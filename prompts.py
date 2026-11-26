"""Prompts for the three Stage A arms.

plain   LiveCodeBench's generic prompt (lcb_runner/prompts/code_generation.py): its system message and question template.
decomp  DecompRL's Appendix I prompts, verbatim apart from restoring the backticks and quotes the arXiv HTML mangled.
idea    The same two prompts with the idea's additions, and nothing else changed:
          * decomposition: each docstring ends with a `Uses:` line, functions are listed basics first
            (the project's `# uses:` format, README §10);
          * implementation: the functions it uses are shown already implemented, not as stubs (dependencies in view,
            the one clear win in README §12.7), and it may stop and name a missing helper (MISSING) instead of calling
            something that doesn't exist (the refuse / suspend step);
          * helper: builds a missing function, after which the caller resumes.

Context: DecompRL fills {context} with "the dataset's problem description". LeetCode problems also need their
function name, so the starter code is appended for functional problems, and a stdin/stdout note for stdin ones.
The same context goes to both decomposition arms.
"""

SYSTEM_LCB = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
              "generate a correct Python program that matches the specification and passes all tests.")
FMT_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
             "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, "
             "it reads the inputs, runs the algorithm and writes output to STDOUT.")


def plain_messages(p):
    q = f"### Question:\n{p['content']}\n\n"
    if p["starter_code"]:
        q += f"### Format: {FMT_STARTER}\n```python\n{p['starter_code']}\n```\n\n"
    else:
        q += f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    q += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM_LCB}, {"role": "user", "content": q}]


def context(p):
    if p["testtype"] == "functional":
        return (f"{p['content']}\n\nThe solution must provide this function (LeetCode starter code):\n"
                f"```python\n{p['starter_code'].rstrip()}\n```")
    return f"{p['content']}\n\nThe program reads the input from stdin and writes the answer to stdout."


_EXAMPLE = '''def merge_and_count(left: list[int], right: list[int]) -> tuple[list[int], int]:
    """
    Merges two sorted lists and counts the number of inversions.
    Args:
        left (list[int]): The left sorted subarray.
        right (list[int]): The right sorted subarray.
    Returns:
        tuple[list[int], int]: A tuple containing the merged sorted list and
        the count of inversions found during the merge.{u1}
    """
    pass

def merge_sort_and_count(arr: list[int]) -> tuple[list[int], int]:
    """
    Recursively sorts an array and counts the number of inversions.
    Args:
        arr (list[int]): The array to be sorted and analyzed for inversions.
    Returns:
        tuple[list[int], int]: A tuple containing the sorted array and the
        total count of inversions in the array.{u2}
    """
    pass

def count_inversions(arr: list[int]) -> int:
    """
    Counts the number of inversions in an array using a modified merge sort.
    Args:
        arr (list[int]): The array to be analyzed for inversions.
    Returns:
        int: The total number of inversions in the array.{u3}
    """
    pass'''

_DECOMP = '''To solve complicated reasoning problems, it is often helpful to break them down into components.
Can you help a student of computer science solve a competitive programming problem by
breaking it down into sensible component functions?

## Example problem

An inversion in an array is a pair of elements where the first element is greater than the second element,
and the first element appears before the second element in the array.
For example, in the array [2, 4, 1, 3, 5], the inversions are (2, 1), (4, 1), and (4, 3). The total number of inversions is 3.

Task: Write a function that counts the number of inversions in an array efficiently.
The goal is to achieve this in O(n log n) time complexity.

## Example decomposition into component functions

```python
{example}
```

In this example decomposition of the programming problem, we give the student the idea
to use count the number of inversions by modifying the merge step of the merge-sort algorithm.
We explain clearly and concisely what the component functions are supposed to do by means of
their type signatures and their docstring, but leave the implementation to the student.

Now, let's do the same for the following competitive programming question:

## Student's competitive programming problem to decompose:

{context}

## Instructions

Please decompose the programming problem into component functions like in the example above.
Guidelines:
- Explain all the functions that the student will need to solve the problem without implementing them.
- Annotate each function with a type signature for its inputs and outputs.
- For each function, write a clear and concise docstring describing what the function does.
- Do not implement the functions: put `pass` as their body.
- The functions can call each other but they should "do one thing" only whereever possible.
    Their behavior should be fully described by their docstring.
- If shared state needs to be passed between the functions, describe it in one of their docstrings
    and make it an argument for the functions that need to read or modify it.
{extra}- Enclose the functions in a single code block using triple backticks like so: ```python YOUR CODE HERE ```.
- You can reason about the problem first before starting the code block.'''

_USES_RULES = ('- End each docstring with a line `Uses: name1, name2` naming the component functions it calls, '
               'or `Uses: -` if it calls none of them.\n'
               '- List the functions basics first: each function comes after every function it uses.\n')

_IMPL = '''I'm trying to solve the following code competitive programming problem:

## Problem description:

{context}

## Instructions

I have decomposed the problem into the following component functions:

```python
{decomposition}
```

Please help me implement the function {current_name}.
You can assume the following functions have been implemented correctly and use them without defining them in your code:
{other_names}
Guidelines:
- Enclose the code in triple backticks like so: ```python YOUR CODE HERE```.
- You can reason about the problem first before starting the code block.
- You can add import statements on top if necessary.
{extra}
Now, please implement this

```python
{current_code}
```

by filling in the function body, keeping the signature and docstring.'''

_MISSING_RULE = ('- Functions shown with a body are already written; read them and call them as they are.\n'
                 '- If you need a helper function that is in neither list, do not write code that calls it. Instead '
                 'reply with only one line per missing function, in the form\n'
                 '  MISSING: def name(arguments) -> return_type: what it does\n'
                 '  and nothing else. It will be written first, then you will be asked again.\n')

_HELPER = '''I'm trying to solve the following code competitive programming problem:

## Problem description:

{context}

## Instructions

I have decomposed the problem into the following component functions:

```python
{decomposition}
```

{why}

Please write the function `{name}`. You can call the functions above; do not redefine them.
Guidelines:
- Enclose the code in triple backticks like so: ```python YOUR CODE HERE```.
- You can reason about the problem first before starting the code block.
- You can add import statements on top if necessary.'''


def decomp_messages(p, uses=False):
    u = ("\n    Uses: -", "\n    Uses: merge_and_count, merge_sort_and_count", "\n    Uses: merge_sort_and_count") if uses \
        else ("", "", "")
    ex = _EXAMPLE.format(u1=u[0], u2=u[1], u3=u[2])
    return [{"role": "user", "content": _DECOMP.format(example=ex, context=context(p),
                                                        extra=_USES_RULES if uses else "")}]


def impl_messages(p, decomposition, current_name, other_headers, current_code, idea=False):
    return [{"role": "user", "content": _IMPL.format(
        context=context(p), decomposition=decomposition, current_name=current_name,
        other_names="\n".join(other_headers) if other_headers else "(none)", current_code=current_code,
        extra=_MISSING_RULE if idea else "")}]


def helper_messages(p, decomposition, name, signature=None, caller=None, caller_code=None):
    if signature:                                          # the model asked for it (MISSING)
        why = f"While writing `{caller}`, this helper was found missing:\n\n    {signature}"
    else:                                                  # the checker found a call to a function nobody defined
        why = (f"This implementation of `{caller}` calls `{name}`, which does not exist yet:\n\n"
               f"```python\n{caller_code}\n```")
    return [{"role": "user", "content": _HELPER.format(context=context(p), decomposition=decomposition, why=why,
                                                        name=name)}]
