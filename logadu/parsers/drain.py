import re
import os
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

class LogCluster:
    """Class to store log clusters with the same template"""
    def __init__(self, log_template='', log_ids=None):
        self.logTemplate = log_template
        self.logIDL = log_ids or []

class DrainNode:
    """Node in Drain's prefix tree structure"""
    def __init__(self, children=None, depth=0, token=None):
        self.childD = children or dict()
        self.depth = depth
        self.token = token

class DrainParser:
    """Drain log parser implementation"""
    
    def __init__(self, depth: int = 4, sim_threshold: float = 0.5, 
                 max_children: int = 100, rex: List[str] = None, 
                 keep_para: bool = False):
        """
        Args:
            depth: Depth of the prefix tree (default: 4)
            sim_threshold: Similarity threshold (0-1) (default: 0.5)
            max_children: Maximum children per node (default: 100)
            rex: List of regex patterns for preprocessing
            keep_para: Whether to keep parameters in output
        """
        self.depth = depth - 2  # Adjust for internal nodes
        self.st = sim_threshold
        self.maxChild = max_children
        self.rex = rex or []
        self.keep_para = keep_para
        self.df_log = None
        self.logname = None

    def parse_file(self, input_path: str, output_dir: str, log_format: str) -> None:
        """Parse a log file using Drain algorithm
        
        Args:
            input_path: Path to input log file
            output_dir: Directory to save results
            log_format: Log format string
        """
        self.input_path = input_path
        self.logname = Path(input_path).stem
        self.savePath = output_dir
        self.log_format = log_format
        
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
            
        starttime = datetime.now()
        print(f'Parsing file: {input_path}')
        
        self._load_data(input_path)
        root_node = DrainNode()
        log_clusters = []
        
        for idx, row in tqdm(self.df_log.iterrows(), total=len(self.df_log)):
            log_id = row['LineId']
            log_message = self._preprocess(row['Content']).strip().split()
            
            # Search for matching cluster
            matched_cluster = self._tree_search(root_node, log_message)
            
            if matched_cluster is None:
                # Create new cluster if no match found
                new_cluster = LogCluster(log_template=log_message, log_ids=[log_id])
                log_clusters.append(new_cluster)
                self._add_to_prefix_tree(root_node, new_cluster)
            else:
                # Update existing cluster
                new_template = self._get_template(log_message, matched_cluster.logTemplate)
                matched_cluster.logIDL.append(log_id)
                if ' '.join(new_template) != ' '.join(matched_cluster.logTemplate):
                    matched_cluster.logTemplate = new_template
        
        self._output_result(log_clusters)
        print(f'Parsing done. [Time taken: {datetime.now() - starttime}]')

    def _tree_search(self, root_node: DrainNode, tokens: List[str]) -> Optional[LogCluster]:
        """Search the prefix tree for matching log cluster"""
        seq_len = len(tokens)
        if seq_len not in root_node.childD:
            return None

        parent_node = root_node.childD[seq_len]
        current_depth = 1
        
        for token in tokens:
            if current_depth >= self.depth or current_depth > seq_len:
                break

            if token in parent_node.childD:
                parent_node = parent_node.childD[token]
            elif '<*>' in parent_node.childD:
                parent_node = parent_node.childD['<*>']
            else:
                return None
            current_depth += 1

        return self._fast_match(parent_node.childD, tokens)

    def _add_to_prefix_tree(self, root_node: DrainNode, log_cluster: LogCluster) -> None:
        """Add a new log cluster to the prefix tree"""
        seq_len = len(log_cluster.logTemplate)
        if seq_len not in root_node.childD:
            root_node.childD[seq_len] = DrainNode(depth=1, token=seq_len)
        
        parent_node = root_node.childD[seq_len]
        current_depth = 1
        
        for token in log_cluster.logTemplate:
            # Add to leaf node if reached depth limit
            if current_depth >= self.depth or current_depth > seq_len:
                if isinstance(parent_node.childD, dict):
                    parent_node.childD = [log_cluster]
                else:
                    parent_node.childD.append(log_cluster)
                break

            # Handle token insertion
            if token not in parent_node.childD:
                if not self._has_numbers(token):
                    if '<*>' in parent_node.childD:
                        if len(parent_node.childD) < self.maxChild:
                            new_node = DrainNode(depth=current_depth+1, token=token)
                            parent_node.childD[token] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.childD['<*>']
                    else:
                        if len(parent_node.childD) + 1 < self.maxChild:
                            new_node = DrainNode(depth=current_depth+1, token=token)
                            parent_node.childD[token] = new_node
                            parent_node = new_node
                        elif len(parent_node.childD) + 1 == self.maxChild:
                            new_node = DrainNode(depth=current_depth+1, token='<*>')
                            parent_node.childD['<*>'] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.childD['<*>']
                else:
                    if '<*>' not in parent_node.childD:
                        new_node = DrainNode(depth=current_depth+1, token='<*>')
                        parent_node.childD['<*>'] = new_node
                        parent_node = new_node
                    else:
                        parent_node = parent_node.childD['<*>']
            else:
                parent_node = parent_node.childD[token]

            current_depth += 1

    def _fast_match(self, log_clusters, tokens: List[str]) -> Optional[LogCluster]:
        """Find the most similar log cluster"""
        max_sim = -1
        max_params = -1
        best_cluster = None

        for cluster in log_clusters:
            sim, params = self._sequence_similarity(cluster.logTemplate, tokens)
            if sim > max_sim or (sim == max_sim and params > max_params):
                max_sim = sim
                max_params = params
                best_cluster = cluster

        return best_cluster if max_sim >= self.st else None

    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> tuple:
        """Calculate similarity between two sequences"""
        assert len(seq1) == len(seq2)
        similar_tokens = 0
        param_count = 0

        for t1, t2 in zip(seq1, seq2):
            if t1 == '<*>':
                param_count += 1
                continue
            if t1 == t2:
                similar_tokens += 1

        return float(similar_tokens) / len(seq1), param_count

    def _get_template(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Generate template from two sequences"""
        assert len(seq1) == len(seq2)
        return [w1 if w1 == w2 else '<*>' for w1, w2 in zip(seq1, seq2)]

    def _load_data(self, input_path: str) -> None:
        """Load and preprocess log data"""
        headers, regex = self._generate_logformat_regex(self.log_format)
        self.df_log = self._log_to_dataframe(input_path, regex, headers)

    def _generate_logformat_regex(self, logformat: str) -> tuple:
        """Generate regex pattern from log format string"""
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r' +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += f'(?P<{header}>.*?)'
                headers.append(header)
                
        return headers, re.compile('^' + regex + '$')

    def _log_to_dataframe(self, log_file: str, regex: re.Pattern, headers: list) -> pd.DataFrame:
        """Convert log file to DataFrame"""
        log_messages = []
        linecount = 0
        
        src_df = pd.read_csv(log_file, low_memory=False)
        if 'Content' not in src_df.columns:
            raise ValueError("CSV file must contain a 'Content' column.")
        for line in tqdm(src_df['Content'].astype(str), desc='Loading log file'):
            line_stripped = line.strip()
            try:
                match = regex.search(line_stripped)
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
            except Exception:
                    continue
        
        # with open(log_file, 'r', encoding='utf8', errors='ignore') as fin:
        #     for line in tqdm(fin.readlines(), desc='Loading log file'):
        #         try:
        #             match = regex.search(line.strip())
        #             if match:
        #                 message = [match.group(header) for header in headers]
        #                 log_messages.append(message)
        #                 linecount += 1
        #         except Exception:
        #             continue
        
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = range(1, linecount + 1)
        return logdf

    def _preprocess(self, line: str) -> str:
        """Preprocess log line by applying regex substitutions"""
        for current_rex in self.rex:
            line = re.sub(current_rex, '<*>', line)
        return line

    def _has_numbers(self, s: str) -> bool:
        """Check if string contains numbers"""
        return any(char.isdigit() for char in s)

    def _output_result(self, log_clusters: List[LogCluster]) -> None:
        """Save parsing results to CSV files"""
        templates = [0] * len(self.df_log)
        template_ids = [0] * len(self.df_log)
        df_events = []
        
        for cluster in tqdm(log_clusters, desc='Generating output'):
            template_str = ' '.join(cluster.logTemplate)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            
            for log_id in cluster.logIDL:
                templates[log_id-1] = template_str
                template_ids[log_id-1] = template_id
            
            df_events.append([template_id, template_str, len(cluster.logIDL)])
        
        self.df_log['EventId'] = template_ids
        self.df_log['EventTemplate'] = templates
        
        org_df = pd.read_csv(self.input_path, low_memory=False)
        # make sure the len of org_df is the same as df_log
        if len(org_df) != len(self.df_log):
            raise ValueError("Original log DataFrame length does not match parsed DataFrame length.")
        self.df_log['Content'] = org_df['Content'] if 'Content' in org_df else None
        self.df_log['Label'] = org_df['Label'] if 'Label' in org_df else None
        
        if False:  # Don't Keep parameters option
            self.df_log['ParameterList'] = self.df_log.apply(
                self._get_parameter_list, axis=1
            )
        
        # Save structured logs
        # keep all original cols from 
        self.df_log.to_csv(
            os.path.join(self.savePath, f'{self.logname}_structured.csv'),
            index=False
        )
        
        # Save templates
        pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences']).to_csv(
            os.path.join(self.savePath, f'{self.logname}_templates.csv'),
            index=False
        )

    def _get_parameter_list(self, row) -> List[str]:
        """Extract parameters from log message"""
        if not self.keep_para:
            return []
            
        template = re.sub(r"<.{1,5}>", "<*>", str(row["EventTemplate"]))
        if "<*>" not in template:
            return []
        
        template = re.sub(r'([^A-Za-z0-9])', r'\\\1', template)
        template = re.sub(r' +', r'\\s+', template)
        template = "^" + template.replace("\<\*\>", "(.*?)") + "$"
        
        params = re.findall(template, row["Content"])
        params = params[0] if params else ()
        params = list(params) if isinstance(params, tuple) else [params]
        
        return [p.strip() for p in params if p.strip()]