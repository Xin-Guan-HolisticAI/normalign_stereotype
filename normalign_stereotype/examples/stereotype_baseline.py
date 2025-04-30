from normalign_stereotype.core._llm_tools import ConfiguredLLM, BulletLLM, StructuredLLM

prompt = """
Given that "A stereotype from a statement is a false generalization of generalized belief that the individuals from the target group from the generalized belief does not possess the attribute from the generalized belief. "

Decide if "Russians are born in Russia".

###Output Format:

["("Yes" or "No"): Explanation(what generalized belief is implied in the statement; why the corresponding generalized belief is a stereotype; if it is a false generalization, how the individuals (with short descriptions) make the generalizations false?)]
- The "short phrase key summary" must be in **exactly** : "Yes" or "No"
- make sure the explanation is self-contained and can be understood without reference to the context.
"""

llm = ConfiguredLLM("deepseek-r1-distill-qwen-1.5b")

response = llm.invoke(
    prompt
)

print(response)