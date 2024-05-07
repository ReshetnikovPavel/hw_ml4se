from datasets import load_dataset

dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
test = dataset['test']#.filter(lambda example: example['label'])

for i in range(10):
    print(i)
    func1 = test['func1'][i]
    func2 = test['func2'][i]
    label = test['label'][i]
    print(func1)
    print(func2)
    print(label)
    print()
