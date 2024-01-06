from prompt_generator import *
from data_utils import *


# Define the main function to decode the answer for a medical question
def fully_decode(qid, realqid, question, options, gold_answer, handler, args, dataobj):
    """_summary_: 
    
    这个 函数定义了一个流程，以系统化的方式使用语言模型（大概率是类似GPT这样的模型）来处理和回答医学多项选择题。
    该函数利用一系列预定义的步骤和检索到的信息，通过不同的方法尝试确定最佳答案。这个过程涉及到多个领域专家的角色模拟，以及对问题和选项的分析。

    Args:
        - qid        : (_type_): _description_
        - realqid    : (_type_): _description_
        - question   : (_type_): _description_
        - options    : (_type_): _description_
        - gold_answer: (_type_): _description_
        - handler    : (_type_): _description_
        - args       : (_type_): _description_
            args.method: 这个参数决定了选择 什么方法来处理问题 & 生成答案的策略。
        - dataobj    : (_type_): _description_

    Returns:
        - : (_type_): _description_
    """
    # 定义一些变量来存储中间结果
    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []

    if args.method == "base_direct": # 仅仅使用语言模型来生成答案 (Direct Method)
        direct_prompt = get_direct_prompt(question, options)
        output = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, max_tokens=50, system_role="")
        ans, output = cleansing_final_output(output) # 清理输出以获得最终答案
    elif args.method == "base_cot": # 使用 逐步提示 (CoT) 来生成答案
        cot_prompt = get_cot_prompt(question, options)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output(output) # 清理输出以获得最终答案
    else: # 使用 MedAgents 来生成答案 (这里的代码是最重要的)
        # (1) 获取 question domains
        question_classifier, prompt_get_question_domain = get_question_domains_prompt(question)
        raw_question_domain = handler.get_output_multiagent(user_input=prompt_get_question_domain, temperature=0, max_tokens=50, system_role=question_classifier)
        if raw_question_domain == "ERROR.":
            raw_question_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_QD)])
        question_domains = raw_question_domain.split(":")[-1].strip().split(" | ")

        # (2) 获取 option domains
        options_classifier, prompt_get_options_domain = get_options_domains_prompt(question, options)
        raw_option_domain = handler.get_output_multiagent(user_input=prompt_get_options_domain, temperature=0, max_tokens=50, system_role=options_classifier)
        if raw_option_domain == "ERROR.":
            raw_option_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_OD)])
        options_domains = raw_option_domain.split(":")[-1].strip().split(" | ")

        # (3) 获取 question analysis
        tmp_question_analysis = []
        for _domain in question_domains:
            question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(question, _domain)
            raw_question_analysis = handler.get_output_multiagent(user_input=prompt_get_question_analysis, temperature=0, max_tokens=300, system_role=question_analyzer)
            tmp_question_analysis.append(raw_question_analysis)
        question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')

        # (4) 获取 option analysis
        tmp_option_analysis = []
        for _domain in options_domains:
            option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(question, options, _domain, question_analyses)
            raw_option_analysis = handler.get_output_multiagent(user_input=prompt_get_options_analyses, temperature=0, max_tokens=300, system_role=option_analyzer)
            tmp_option_analysis.append(raw_option_analysis)
        option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')

        if args.method == "anal_only": # 使用问题和选项的分析来直接推导出答案。
            answer_prompt = get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses)
            output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
            ans, output = cleansing_final_output(output)
        else:
            # get synthesized report
            q_analyses_text = transform_dict2text(question_analyses, "question", question)
            o_analyses_text = transform_dict2text(option_analyses, "options", options)
            synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(q_analyses_text, o_analyses_text)
            raw_synthesized_report = handler.get_output_multiagent(user_input=prompt_get_synthesized_report, temperature=0, max_tokens=2500, system_role=synthesizer)
            syn_report = cleansing_syn_report(question, options, raw_synthesized_report)

            if args.method == "syn_only": # 使用问题和选项的分析来直接推导出答案。
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report)
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
                ans, output = cleansing_final_output(output)
            elif args.method == "syn_verif": # 通过一系列的共识和修订过程来验证和改进 synthesis report 直至达到满意的结果。
                all_domains = question_domains + options_domains

                syn_repo_history = [syn_report]

            
                hasno_flag = True   # default value: in order to get into the while loop
                num_try = 0

                while num_try < args.max_attempt_vote and hasno_flag:
                    domain_opinions = {}    # 'domain' : 'yes' / 'no'
                    revision_advice = {}
                    num_try += 1
                    hasno_flag = False
                    # hold a meeting for all domain experts to vote and gather advice if they do not agree
                    for domain in all_domains:
                        voter, cons_prompt = get_consensus_prompt(domain, syn_report)  # 让不同领域的专家对综合报告进行投票
                        raw_domain_opi = handler.get_output_multiagent(user_input=cons_prompt, temperature=0, max_tokens=30, system_role=voter)
                        domain_opinion = cleansing_voting(raw_domain_opi)   # "yes" / "no"
                        domain_opinions[domain] = domain_opinion 
                        if domain_opinion == "no":
                            advice_prompt = get_consensus_opinion_prompt(domain, syn_report) # 收集不同意见的专家的修订建议
                            advice_output = handler.get_output_multiagent(user_input=advice_prompt, temperature=0, max_tokens=500, system_role=voter)
                            revision_advice[domain] = advice_output 
                            hasno_flag = True
                    if hasno_flag:
                        revision_prompt = get_revision_prompt(syn_report, revision_advice) # 根据修订建议更新综合报告
                        revised_analysis = handler.get_output_multiagent(user_input=revision_prompt, temperature=0, max_tokens=2500, system_role="")
                        syn_report = cleansing_syn_report(question, options, revised_analysis)
                        revision_history.append(revision_advice)
                        syn_repo_history.append(syn_report)
                    vote_history.append(domain_opinions)
                
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report)
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
                ans, output = cleansing_final_output(output)
                        


    data_info = {
        'question': question,
        'options': options,
        'pred_answer': ans,
        'gold_answer': gold_answer,
        'question_domains': question_domains,
        'option_domains': options_domains,
        'question_analyses': question_analyses,
        'option_analyses': option_analyses,
        'syn_report': syn_report,
        'vote_history': vote_history,
        'revision_history': revision_history,
        'syn_repo_history': syn_repo_history,
        'raw_output': output
    }
    
    return data_info

