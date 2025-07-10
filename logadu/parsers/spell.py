import re
import regex as re
import os
import pandas as pd
import hashlib
from datetime import datetime
import string
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path

class LCSObject:
    """Class to store log groups with the same template"""
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        self.logIDL = logIDL or []

class Node:
    """Node in prefix tree data structure"""
    def __init__(self, token='', templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()

class SpellParser:
    """Spell log parser implementation"""
    
    def __init__(self, tau: float = 0.5, rex: List[str] = None, keep_para: bool = True):
        """
        Args:
            tau: Similarity threshold (0-1)
            rex: List of regex patterns for preprocessing
            keep_para: Whether to keep parameters in output
        """
        self.tau = tau
        self.rex = rex or []
        self.keep_para = keep_para
        self.df_log = None
        self.logname = None

    def parse_file(self, input_path: str, output_dir: str, log_format: str) -> None:
        """Parse a log file using Spell algorithm
        
        Args:
            input_path: Path to input log file
            output_dir: Directory to save results
            log_format: Log format string
        """
        self.logname = Path(input_path).stem
        self.savePath = output_dir
        self.logformat = log_format
        
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
            
        starttime = datetime.now()
        print(f'Parsing file: {input_path}')
        
        self._load_data(input_path)
        rootNode = Node()
        logCluL = []
        
        punc = re.sub('[<*>]', '', string.punctuation)
        
        for idx, line in tqdm(self.df_log.iterrows(), total=len(self.df_log)):
            logID = line['LineId']
            logmessageL = list(filter(lambda x: x.strip() != '', 
                                   re.split(f'[{punc}]', self._preprocess(line['Content']))))
            constLogMessL = [w for w in logmessageL if w != '<*>']

            matchCluster = self._prefix_tree_match(rootNode, constLogMessL, 0)
            
            if matchCluster is None:
                matchCluster = self._simple_loop_match(logCluL, constLogMessL)
                
                if matchCluster is None:
                    matchCluster = self._lcs_match(logCluL, logmessageL)
                    
                    if matchCluster is None:
                        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])
                        logCluL.append(newCluster)
                        self._add_seq_to_prefix_tree(rootNode, newCluster)
                    else:
                        newTemplate = self._get_template(
                            self._lcs(logmessageL, matchCluster.logTemplate),
                            matchCluster.logTemplate
                        )
                        if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                            self._remove_seq_from_prefix_tree(rootNode, matchCluster)
                            matchCluster.logTemplate = newTemplate
                            self._add_seq_to_prefix_tree(rootNode, matchCluster)
            
            if matchCluster:
                matchCluster.logIDL.append(logID)
                
        self._output_result(logCluL)
        print(f'Parsing done. [Time taken: {datetime.now() - starttime}]')

    def _lcs(self, seq1, seq2):
        """Longest common subsequence between two sequences"""
        lengths = [[0 for _ in range(len(seq2)+1)] for _ in range(len(seq1)+1)]
        
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        result = []
        len1, len2 = len(seq1), len(seq2)
        while len1 != 0 and len2 != 0:
            if lengths[len1][len2] == lengths[len1-1][len2]:
                len1 -= 1
            elif lengths[len1][len2] == lengths[len1][len2-1]:
                len2 -= 1
            else:
                assert seq1[len1-1] == seq2[len2-1]
                result.insert(0, seq1[len1-1])
                len1 -= 1
                len2 -= 1
        return result

    def _simple_loop_match(self, logClustL, seq):
        for logClust in logClustL:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)
            if all(token in token_set or token == '<*>' for token in logClust.logTemplate):
                return logClust
        return None

    def _prefix_tree_match(self, parentn, seq, idx):
        """Iterative prefix tree matching to avoid recursion limits"""
        retLogClust = None
        length = len(seq)
        
        # Use stack to simulate recursion (node, current_index)
        stack = [(parentn, idx)]
        
        while stack:
            current_node, current_idx = stack.pop()
            
            # Process current level
            for i in range(current_idx, length):
                token = seq[i]
                
                # Check for matching child node
                if token in current_node.childD:
                    child_node = current_node.childD[token]
                    
                    # Check if this is a log cluster node
                    if child_node.logClust is not None:
                        const_tokens = [w for w in child_node.logClust.logTemplate if w != '<*>']
                        if float(len(const_tokens)) >= self.tau * length:
                            return child_node.logClust
                    else:
                        # Push next level to stack
                        stack.append((child_node, i + 1))
                        break  # Move to next level
                else:
                    break  # No match at this level
        
        return retLogClust

    #for each seq, find the corresponding log template using LCS
    def _lcs_match(self, logClustL, seq):
        retLogClust = None

        maxLen = -1
        maxlcs = []
        maxClust = None
        set_seq = set(seq)
        size_seq = len(seq)
        for logClust in logClustL:
            set_template = set(logClust.logTemplate)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self._lcs(seq, logClust.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(logClust.logTemplate) < len(maxClust.logTemplate)):
                maxLen = len(lcs)
                maxlcs = lcs
                maxClust = logClust

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        return retLogClust

    def _get_template(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        if i < len(seq):
            retVal.append('<*>')
        return retVal

    def _add_seq_to_prefix_tree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
                # Do not Match
            else:
                parentn.childD[tokenInSeq] = Node(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.logClust is None:
            parentn.logClust = newCluster

    def _remove_seq_from_prefix_tree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def _output_result(self, logClustL):
        print("output result", self.savePath)
        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []

        for logclust in tqdm(logClustL):
            template_str = ' '.join(logclust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logid in logclust.logIDL:
                templates[logid - 1] = template_str
                ids[logid - 1] = eid
            df_event.append([eid, template_str, len(logclust.logIDL)])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self._get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logname + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False)

    def _print_tree(self, node, dep):
        pStr = ''
        for i in range(len(dep)):
            pStr += '\t'

        if node.token == '':
            pStr += 'Root'
        else:
            pStr += node.token
            if node.logClust is not None:
                pStr += '-->' + ' '.join(node.logClust.logTemplate)
        print(pStr + ' (' + str(node.templateNo) + ')')

        for child in node.childD:
            self._print_tree(node.childD[child], dep + 1)


    def _load_data(self, input_path: str) -> None:
        """Load and preprocess log data from file
        
        Args:
            input_path: Path to the input log file
        """
        headers, regex = self._generate_logformat_regex(self.logformat)
        self.df_log = self._log_to_dataframe(input_path, regex, headers)

    def _preprocess(self, line: str) -> str:
        """Preprocess log line by applying regex substitutions
        
        Args:
            line: Raw log line
            
        Returns:
            str: Preprocessed line with variables replaced by <*>
        """
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def _log_to_dataframe(self, log_file: str, regex: re.Pattern, headers: list) -> pd.DataFrame:
        """Transform log file to pandas DataFrame
        
        Args:
            log_file: Path to log file
            regex: Compiled regex pattern for parsing
            headers: List of column headers
            
        Returns:
            pd.DataFrame: Parsed log data
        """
        log_messages = []
        linecount = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as fin:
            for line in tqdm(fin.readlines(), desc='Loading log file'):
                try:
                    line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line.strip())
                    match = regex.search(line)
                    if match:
                        message = [match.group(header) for header in headers]
                        log_messages.append(message)
                        linecount += 1
                except Exception as e:
                    continue
        
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = range(1, linecount + 1)
        return logdf

    def _generate_logformat_regex(self, logformat: str) -> tuple:
        """Generate regular expression to split log messages
        
        Args:
            logformat: Log format string containing fields like <Date>, <Content>
            
        Returns:
            tuple: (headers, compiled regex pattern)
        """
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

    def _get_parameter_list(self, row):
        template_regex = re.sub(r"\s<.{1,5}>\s", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'[^A-Za-z0-9]+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        parameter_list = [para.strip(string.punctuation).strip(' ') for para in parameter_list]
        return parameter_list

