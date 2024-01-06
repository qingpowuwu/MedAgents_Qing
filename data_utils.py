import os
import jsonlines
import re
import nltk
nltk.download('punkt') # 下载 Punkt Tokenizer  pre-trained model, 
from nltk.tokenize import sent_tokenize
# sent_tokenize 能够处理各种句子结束符号，如句号、问号和感叹号，并能在大多数情况下正确地识别句子边界。
from rouge_score import rouge_scorer

#%%

class MyDataset:
    def __init__(self, 
                 split, 
                 args, 
                 eval_only=False, 
                 traindata_obj=None):
        """_summary_: 
        
        这个 MyDataset 类是为了处理和管理一个特定数据集而设计的。它包含多个方法，用于加载数据、获取特定索引的数据项、计算评估指标（如ROUGE得分和准确度）等。

        Args:
            - split        : (str) : 用来指定 放到 'train' 目录 或者 'test' 目录
                    ex: split = 'train' or split = 'test'
            - args         : (obj) : 包含配置信息的对象。
            - eval_only    : (bool): 是否仅用于评估
                                                        [Defaults to False]
            - traindata_obj: (obj): 可选的训练数据对象.
                                                        [Defaults to None]
        """
        
        # 根据 args 设置实例变量
        # hasattr(object, name)函数：如果对象有该属性返回 True，否则返回 False。
        if hasattr(args, 'start_pos'):
            self.start_pos = args.start_pos
        if hasattr(args, 'end_pos'):
            self.end_pos = args.end_pos
        if hasattr(args, 'model_name'):
            self.model_name = args.model_name
        self.dataset_name = args.dataset_name # 'MedQA'
        self.dir_path     = args.dataset_dir  # './datasets/MedQA/'
        self.split        = split # 'train' or 'test'
        self.load() # load dataset -> load data in self.data
        # load answers -> self.choice_ref / self.ref
        
        # 根据数据集名称构建引用答案
        if args.dataset_name == 'MedQA':
            self.build_choice_ref_MedQA()
        elif args.dataset_name == 'MedMCQA' or 'MMLU' in args.dataset_name:
            self.build_choice_ref_MedMCQA()
        elif args.dataset_name == 'PubMedQA':
            self.build_choice_ref_MedMCQA()
        elif args.dataset_name == 'MedicationQA':
            self.build_ref()
        

    def load(self): # load dataset -> self.data
        """ 加载数据集
            通过读取 JSONL 文件来加载数据，文件路径是由数据集目录和分割名组合而成的。
        """
        filename = os.path.join(self.dir_path, self.split + '.jsonl') # /Users/qingpowuwu/Library/Mobile Documents/com~apple~CloudDocs/1_Research/23_PDE_Agents/18_MedAgents_Qing/datasets/MedQA/test.jsonl
        self.data = []
        with open(filename) as f:
            for item in jsonlines.Reader(f):
                self.data.append(item)

    def get_by_idx(self, idx):
        """ 根据索引获取数据集中的单个数据项。 

        Args:
            - idx:            (int): 

        Returns:
            - self.data[idx]: (str): 
        """
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def build_ref(self):
        """
        为MedicationQA数据集构建引用答案。
        
        此方法遍历数据集中的每一项，提取出标准答案，并将其存储在一个 list self.ref 中。
        每个数据项的标准答案被存储在一个字典中，包含答案文本和数据项的唯一标识符。
        """
        self.ref = [] # 初始化1个 list, 这个 list 将用于存储每个数据项的引用答案。
        for i in range(len(self)): # 这里的 self 相当于调用了 __len__(MyDataset实例) 方法
            item = self.get_by_idx(i)
            self.ref.append({'answers': {'text': item['answer']}, 
                             'id': i})
    
    def build_choice_ref_MedQA(self):
        """
        为MedQA数据集构建引用答案。
        """
        self.choice_ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.choice_ref.append({
                'answers': {'text': item['answer'],'choice': item['answer_idx']}, 
                'options': item['options'], 
                'type': item['meta_info'],
                'id': i})

    def build_choice_ref_MedMCQA(self):
        """
        为MedMCQA数据集构建引用答案。
        """
        self.choice_ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.choice_ref.append({
                'answers': {'text': item['answer'],
                'choice': item['answer_idx']}, 
                'options': item['options'], 
                'id': i})

    # 计算评估指标的方法
    def compute_rougescore(self, preds):
        """
        计算ROUGE得分。
        """
        sum_score = 0.0
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i, answer in enumerate(preds):
            correct_answer = self.ref[i]['answers']['text']
            # correct_answer = correct_answer.replace('\n', ' ')
            score = scorer.score(answer, correct_answer)
            sum_score += score['rouge1'].fmeasure
            # print(f'id: {i}, answer: {answer}, correct answer: {correct_answer}, rouge1 score: {score["rouge1"].fmeasure}')
            # print(score)
            # break
        return sum_score / len(preds)

    def compute_accuracy(self, preds):
        """
        计算准确度。
        """
        # 根据不同的数据集名称，计算准确度
        if 'PubMedQA' in self.dir_path:
            correct_num = 0.0
            all_num = 0.0
            for i, answer in enumerate(preds):
                all_num += 1
                correct_choice = self.choice_ref[i]['answers']['choice']
                correct_answer = self.choice_ref[i]['answers']['text']
                if answer == correct_choice or correct_answer in answer:
                    correct_num += 1
                # print(f"id: {i}, choice: {answer}, correct choice: {correct_choice}")
            print(f"correct_num: {correct_num}, all_num: {all_num}")
            return correct_num / all_num
        elif 'MedQA' in self.dir_path:
            correct_num = {'step1': 0.0, 'step2&3': 0.0, 'all': 0.0}
            all_num = {'step1': 0.0, 'step2&3': 0.0, 'all': 0.0}
            for i, answer in enumerate(preds):
                # choice = answer[:3]
                answer = answer.strip()
                all_num['all'] += 1
                correct_choice = self.choice_ref[i]['answers']['choice']
                correct_answer = self.choice_ref[i]['answers']['text']
                type = self.choice_ref[i]['type']
                all_num[type] += 1
                if answer == correct_choice or (correct_choice in answer and answer != 'ERROR') or correct_answer in answer:
                    correct_num[type] += 1
                    correct_num['all'] += 1
                # print(f"id: {i}, choice: {answer}, correct choice: {correct_choice}")
            print(f"correct_num: {correct_num}, all_num: {all_num}")
            return [correct_num[key] / all_num[key] for key in ['step1', 'step2&3', 'all']]
        elif 'MedMCQA' in self.dir_path or 'MMLU' in self.dir_path:
            correct_num = 0.0
            all_num = 0.0
            for i, answer in enumerate(preds):
                # choice = answer[:3]
                all_num += 1
                correct_choice = self.choice_ref[i]['answers']['choice']
                correct_answer = self.choice_ref[i]['answers']['text']
                if answer == correct_choice or correct_answer in answer:
                    correct_num += 1
                # print(f"id: {i}, choice: {answer}, correct choice: {correct_choice}")
            print(f"correct_num: {correct_num}, all_num: {all_num}")
            return correct_num / all_num

#%%

def remove_incomplete_sentence(text):
    """ 移除文本中的不完整句子。

    Args:
        - text: (str): 原始文本。

    Returns:
        - text: (str): 清理后的文本，其中不完整的句子被移除。
    """
    sentences = sent_tokenize(text) # 使用 nltk 库的 sent_tokenize 函数将输入的文本 text 分割成句子。'sent_tokenize' 函数能够识别文本中的句子边界，将文本分割成一个句子列表。
    if len(sentences) > 1 and sentences[-1][-1] != '.': # 如果最后一个句子是不完整的，移除它，并将其余句子重新连接起来。
        return ' '.join(sentences[:-1]) + '.'   # 移除最后一个句子
    else:
        return text # 如果文本只有一个句子或最后一个句子是完整的，保留原始文本
    
# 这个函数的主要作用是确保自动生成的文本中不含有由于截断等原因产生的不完整句子，从而提高文本的整体质量和可读性。通过检查并移除不完整的句子，它帮助确保输出的文本是清晰且连贯的。

# 测试用例
text_example_1 = "This is a complete sentence. This is another complete sentence."
text_example_2 = "This is a complete sentence. This one is not complete"
text_example_3 = "Only one sentence here but incomplete"

# 测试函数
test_results = []
test_results.append(remove_incomplete_sentence(text_example_1))
test_results.append(remove_incomplete_sentence(text_example_2))
test_results.append(remove_incomplete_sentence(text_example_3))

print('test_results = ', test_results) # ['This is a complete sentence. This is another complete sentence.', 'This is a complete sentence..', 'Only one sentence here but incomplete']

#%% 清洗 llm 给的 analyses 的函数 => 把 analyses 转换成 固定的形式：Reporti \n Options: xxx \n Domain: xxx \n Analysis: xxx

def cleansing_analysis(analyses, domains, type):
    """ 清洗和整理来自不同领域的分析结果。

    Args:
        - analyses: (list): 一个包含不同领域分析的 list。
        - domains : (list): 对应分析内容的领域列表。
        - type    : (str) : 分析的类型（如 'question' 或 'options'）。

    Returns:
        - analysis: (dict): 清洗和整理后的分析结果字典。
    """
    analysis = {} # 初始化一个空字典用于存储清洗后的分析结果
    
     # 遍历分析列表，处理每个分析项
    for i, item in enumerate(analyses): # item: 当前的分析内容
        if item == "ERROR.": # 如果当前的分析内容为 "ERROR."，表示在生成分析时出现了问题
            item = f"There is no analysis for this {type}."  # 设置一个默认消息表明该类型没有分析结果
        item = remove_incomplete_sentence(item)  # 移除可能的不完整句子，提高分析内容的质量
        if "as an ai language model" in item.lower(): #  检查分析内容中是否包含特定的字符串（"as an ai language model"）。这可能是一个指示，表明文本是由AI语言模型生成的自我介绍或说明。
            end_index = item.lower().find("as an ai language model")+len("as an ai language model") # 截取掉包含该字符串及其之前的部分，并清理前后的空白字符和逗号，以保证文本的清晰度。
            item = item[end_index:].strip().strip(',').strip() 
        analysis[domains[i]] = item  # 将清洗后的分析内容与对应的领域（domains[i]）关联，并存储在 analysis 字典中。
    
    # 返回包含所有清洗后分析的字典
    return analysis

# 在这个函数中，对每个分析项进行了细致的清洗和处理，以确保它们的质量和准确性。这样做可以帮助确保最终提供给用户或用于决策的信息是可靠和有用的。

# 测试用例
analyses_example = [
    "The patient shows signs of improvement... as an AI language model, I suggest...",
    "ERROR.",
    "Normal blood pressure levels."
]

domains_example = ["Cardiology", "Neurology", "General Medicine"]

# 测试函数
cleaned_analyses = cleansing_analysis(analyses_example, domains_example, "question")

print('cleaned_analyses = ', cleaned_analyses)

#%% 清洗 llm 给的 output 的函数 => 用来提取  'A', 'B', 'C', 'D', 'E'字符

def cleansing_syn_report(question, options, raw_synthesized_report):
    """_summary_: 清洗合成报告的输出函数，用于提取和清洁由 llm 生成的合成报告。

    Args:
        - question              : (str): 相关问题(question)的文本。
                ex: question = "What is the treatment for a viral infection?"
        - options               : (str): 相关选项(options)的文本。
                ex: options = "A) Antibiotics B) Antiviral medication C) Pain relievers"
        - raw_synthesized_report: (str): 原始的合成报告文本。
                ex: raw_synthesized_report = "Key Knowledge: Viral infections cannot be treated with antibiotics. " \
                     "Total Analysis: The best treatment is antiviral medication, " \
                     "though pain relievers can help with symptoms."

    Returns:
        - final_syn_repo: (str): 清洗和格式化后的综合报告文本。
    """

    tmp = raw_synthesized_report.split("Total Analysis:") # 这一行将原始的合成报告文本以 "Total Analysis:" 为分隔符分割。这样做是为了将报告中的 "Total Analysis" 部分与其他内容分开。
    total_analysis_text = tmp[1].strip() # 这一行将分割后的第二部分 "Total Analysis" 作为 "total_analysis_text" , 并且用 strip()去除 空白的自字符
    
    # 如果存在 "Key Knowledge"，则进一步提取这部分的内容：
    if "Key Knowledge" in tmp:
        key_knowledge_text = tmp[0].split("Key Knowledge:")[-1].strip() # 获取 "Key Knowledge" 部分的文本。这里使用了再次分割并取最后一个元素的方式来提取文本，并去除周围的空白字符。
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Key Knowledge: {key_knowledge_text} \n" \
            f"Total Analysis: {total_analysis_text} \n" # 将 "Key Knowledge" 和 "Total Analysis" 部分的文本与问题和选项一起构造成最终的综合报告。
    # 如果没有 "Key Knowledge" 部分，只构造包含问题、选项和总体分析的报告
    else:
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Total Analysis: {total_analysis_text} \n"
    # 返回最终构造好的综合报告字符串。
    return final_syn_repo


# 总的来说，这个函数用于将原始的综合报告文本清洗和重构成更加规范和易于阅读的格式。这对于在进行决策或进一步分析时提供清晰、结构化的信息非常有用。通过这种方式，可以确保重要的信息部分（如 "Key Knowledge" 和 "Total Analysis"）被正确识别和展示。

# 测试用例
question_example = "What is the treatment for a viral infection?"
options_example = "A) Antibiotics B) Antiviral medication C) Pain relievers"
raw_report_example = "Key Knowledge: Viral infections cannot be treated with antibiotics. " \
                     "Total Analysis: The best treatment is antiviral medication, " \
                     "though pain relievers can help with symptoms."

# 测试函数
cleaned_report = cleansing_syn_report(question_example, options_example, raw_report_example)

print("cleaned_report = \n", cleaned_report)


#%% 清洗 llm 给的 output 的函数 => 用来提取  'A', 'B', 'C', 'D', 'E'字符

def cleansing_final_output(output):
    """_summary_: 清洗输出函数，用于提取和清洁由 llm 生成的最终答案。

    Args:
        - output: (_type_): 模型生成的 原始 string。

    Returns: 
        - ans    : (str): 提取出的答案选项，如果没有找到有效答案则为空字符串。
            ex1: ans = 'A'
            ex2: ans = 'B'
        - output: (str) : 原始的输出字符串，用于后续的验证或记录。
            ex1: output = 'The correct answer is: C'
            ex2: output = 'Option B seems most plausible.'
            
    """
    # 开始一个 try 语句块，尝试执行一组操作，如果出现错误，则执行 except 块中的代码。
    try:
        ans = output.split(":")[-1] # 将 output 字符串按冒号（:）分割(通过 xxx.split()方法)，并尝试取 最后一个元素作为潜在的答案(通过[-1])。
        ans = re.findall(r'A|B|C|D|E', ans) # 使用正则表达式查找字符串 ans 中的所有 'A', 'B', 'C', 'D', 'E'字符
        if len(ans) == 0: # 如果没有找到任何匹配项，则 ans 为空字符串
            ans = ""
        else: #  如果找到了匹配项，取匹配列表的第一个元素作为 ans
            ans = ans[0] 
    except:
        ans = re.findall(r'A|B|C|D|E', ans)
        if len(ans) == 0: # 如果没有找到任何匹配项，则 ans 为空字符串
            ans = ""
        else:
            ans = ans[-1] # 如果找到匹配项，选择最后一个作为 ans
            
    # 返回处理过的答案(ans) 和 原始输出(output)
    return ans, output 

# 测试用例
output_example_1 = "The correct answer is: C"
output_example_2 = "Option B seems most plausible."
output_example_3 = "I'm not sure, but it might be: D or A"
output_example_4 = "This is an invalid output without any option"

# 测试函数
test_results = []
test_results.append(cleansing_final_output(output_example_1)) # ('C', 'The correct answer is: C')
test_results.append(cleansing_final_output(output_example_2)) # ('B', 'Option B seems most plausible.')
test_results.append(cleansing_final_output(output_example_3)) # ('D', "I'm not sure, but it might be: D or A")
test_results.append(cleansing_final_output(output_example_4)) # ('', 'This is an invalid output without any option')]

print('test_results = ', test_results)

#%% 清洗 llm 给的 output 的函数 => 用来提取 yes/no 的答案

def cleansing_voting(output):
    """_summary_: 清洗投票结果的输出函数，用于提取处理过的 yes/no 的回答。

    Args:
        - output: (str): 从模型或其他源获得的原始输出字符串。

    Returns:
        - ans:    (str): 提取并处理过的答案，'yes' 或 'no'。如果在输出中没有明确找到，则默认为 'yes'。
    """
    output = output.lower()
    ans = re.findall(r'yes|no', output) # 使用正则表达式查找所有出现的 'yes' 或 'no' 字符
    if len(ans) == 0: # 如果没有找到任何匹配项，则 ans 为空字符串
        ans = "yes" # 默认返回 'yes'
    else:
        ans = ans[0] # 否则，返回找到的第一个 'yes' 或 'no'
        
    # 返回处理过的答案(ans)
    return ans

# 测试用例
output_example_1 = "I think the answer should be YES"
output_example_2 = "no, I don't agree with that"
output_example_3 = "This is unclear"
output_example_4 = "Yes, I think that's correct"

# 测试函数
test_results = []
test_results.append(cleansing_voting(output_example_1)) # 'yes'
test_results.append(cleansing_voting(output_example_2)) # 'no'
test_results.append(cleansing_voting(output_example_3)) # 'yes'
test_results.append(cleansing_voting(output_example_4)) # 'yes'

print('test_results = ', test_results) # ['yes', 'no', 'yes', 'yes']

#%% 清洗 llm 给的 analyses 的函数 => 把 analyses 转换成 固定的形式：Reporti \n Options: xxx \n Domain: xxx \n Analysis: xxx

def transform_dict2text(analyses, type, content):
    """_summary_: 将分析结果字典转换为格式化的文本报告。

    Args:
        - analyses: (dict): 包含各个 domain expert 给的分析内容的字典。
                            key 为 domain，value 为 analysis 的内容
                    ex: analyses = {'domain1': 'analysis1', 
                                    'domain2': 'analysis2', ...}
        - type    : (str) : 转换类型，'question' 或 'options'，影响报告的格式。
                    ex: type = 'question'
        - content : (str) : 分析内容相关的问题或选项。对于 "question" 类型，这可能是一个具体的医学问题；对于 "options" 类型，这可能是多项选择题的选项。
                    ex: content = 'What is the most likely diagnosis?'

    Returns:
        - report (str): 一个包含所有分析的格式化字符串，这个字符串可以用于后续的展示或处理。
                    ex: report_question = 
                                    '''
                                        Report0 
                                        Question: A 55-year-old man with chest pain. 
                                        Domain: Cardiology 
                                        Analysis: The patient's heart rate is abnormal. 

                                        Report1 
                                        Question: A 55-year-old man with chest pain. 
                                        Domain: Neurology 
                                        Analysis: No signs of neurological damage. 
                                    '''
    """
    # 如果类型是 'question'，则按照问题的分析格式生成报告
    if type == "question":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
             # 为每个领域和对应的分析添加一段文本到报告中
            report += f"Report{i} \n" \
                f"Question: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1

    # 如果类型是 'options'，则按照选项的分析格式生成报告
    elif type == "options":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            # 为每个领域和对应的分析添加一段文本到报告中
            report += f"Report{i}: \n" \
                f"Options: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    # 返回格式化的文本报告
    return report

# 这个函数通过整合来自不同专业领域的分析结果，生成一个清晰、易于阅读的文本报告。这对于展示复杂数据或在决策支持系统中提供信息汇总非常有用。

# 测试用例
analyses_example = {
    "Cardiology": "The patient's heart rate is abnormal.",
    "Neurology": "No signs of neurological damage."
}
question_content = "A 55-year-old man with chest pain."
options_content = "A) Aspirin B) Surgery"

# 测试 question 类型
report_question = transform_dict2text(analyses_example, 
                                      "question", 
                                      question_content)
# 测试 option 类型
report_options = transform_dict2text(analyses_example, 
                                     "options", 
                                     options_content)

print("report_question = \n", report_question)
print("report_options  = \n", report_options)

#%%