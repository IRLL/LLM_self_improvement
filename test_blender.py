import llm_blender
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") # load ranker checkpoint
inputs = ["hello, how are you!", "I love you!"]
candidates_texts = [["get out!", "hi! I am fine, thanks!", "bye!"], 
                    ["I love you too!", "I hate you!", "Thanks! You're a good guy!"]]
ranks = blender.rank(inputs, candidates_texts, return_scores=False, batch_size=1)

print("ranks",ranks)