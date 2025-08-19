from swift.llm import sft_main, TrainArguments
result = sft_main(TrainArguments(
    model='Qwen/Qwen3-0.6B',
    train_type='lora',
    dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500'],
    torch_dtype='bfloat16',
))
print("done!!")