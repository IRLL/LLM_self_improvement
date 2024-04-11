import transformers
import torch
import os,tqdm,json

from utils import calculate_classification_metrics,log_method,ClearCache


@log_method
def eval_boolq(model, tokenizer, boolq_eval_path, boolq_eval_result_path):
    with ClearCache():
        model.eval()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=10


        )
        pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
        with open(boolq_eval_path) as obj:
            boolq_data = json.loads(obj.read())

        #print(len(boolq_data))
        boolq_data =boolq_data[:500]

        prompt = """Write a response that appropriately completes answer the question, follow the examples. Your answer should be "True" or "False".

        ### Example 1:
        Passage: 
        The Vampire Diaries, an American supernatural drama, was renewed for an eighth season by The CW on March 11, 2016. On July 23, 2016, the CW announced that the upcoming season would be the series' last and would consist of 16 episodes. The season premiered on October 21, 2016 and concluded on March 10, 2017.

        Question:
        will there be a season 8 of vampire diaries?

        Answer:
        True

        ### Example 2:
        Passage: 
        This is the list of U.S. states that have participated in the Little League World Series. As of the 2018 LLWS, eight states had never reached the LLWS: Alaska, Colorado, Kansas, North Dakota, Utah, Vermont, Wisconsin, and Wyoming; additionally, the District of Columbia has never reached the LLWS.

        Question:
        has wisconsin ever been in the little league world series?

        Answer:
        False

        ### Task:
        Passage:
        {passage}

        Question:
        {question}

        Answer:
        """

        predictions = []
        labels = []
        for idx, item in enumerate(boolq_data): 
            full_prompt = prompt.format(question=item['question'], passage=item['passage'])
            result = pipeline(full_prompt)
            #print("full_prompt",full_prompt)
            truncated = result[0]['generated_text'][len(full_prompt):].strip()
            #print(truncated)
            if "false" in truncated or "False" in truncated:
                predictions.append(0)
            else:
                predictions.append(1)
            labels.append(item["answer"])
        # print(predictions, labels)
        metrics = calculate_classification_metrics(predictions, labels)

        with open(boolq_eval_result_path,'w') as obj:
            obj.write(json.dumps(metrics))
        
        del pipeline,labels,predictions,boolq_data


    return metrics