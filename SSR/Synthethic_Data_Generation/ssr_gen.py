from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import jsonlines
import argparse
from tqdm import tqdm
import random
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import os

# Prompt template
llama_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Here are some examples:

Example 1:
{example_1}
Example 2:
{example_2}

{prompt}
"""

def generate_embeddings(data, text_field='input', model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Generates embeddings for the given data using a sentence transformer model.
    """
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Collect texts to embed
    texts = [task[text_field] for task in data]
    
    # Generate embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    
    return embeddings

def cluster_and_sample_data(data, embeddings, sample_memory=200, n_clusters=20):
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    labels = kmeans.fit_predict(embeddings_norm)
    
    # Compute distances to cluster centers
    centric_distances = np.linalg.norm(embeddings_norm - kmeans.cluster_centers_[labels], axis=1)
    
    # Count instances in each cluster
    n_cluster_instances = np.bincount(labels, minlength=n_clusters)
    
    # Determine number of samples per cluster proportionally
    total_instances = len(data)
    clu_sample_num = [max(1, round(sample_memory * count / total_instances)) for count in n_cluster_instances]
    
    # Sample data points closest to cluster centers
    sampled_indices = []
    for clu_idx in range(n_clusters):
        cluster_indices = np.where(labels == clu_idx)[0]
        cluster_distances = centric_distances[cluster_indices]
        num_samples = min(clu_sample_num[clu_idx], len(cluster_indices))
        if num_samples > 0:
            # Get indices of closest points
            closest_indices = cluster_indices[np.argsort(cluster_distances)[:num_samples]]
            sampled_indices.extend(closest_indices)
    
    # Get sampled data
    sampled_data = [data[i] for i in sampled_indices]
    
    return sampled_data

def prepare_prompts(data, template):
    """
    Prepare prompts using the specified template and add two random examples to the instructions.
    """
    prompts = []
    for task in data:
        # Extract the definition from full_prompt
        definition = task.get('full_prompt', '').split('\n\n')[0]
        
        # Randomly select two input-output pairs from the sampled data
        examples = random.sample(data, 2)
        
        # Function to extract context and question
        def extract_context_question(input_text):
            context_marker = "Context:"
            question_marker = "Question:"
            context = ""
            question = ""
            if context_marker in input_text and question_marker in input_text:
                context_start = input_text.find(context_marker) + len(context_marker)
                question_start = input_text.find(question_marker)
                context = input_text[context_start:question_start].strip()
                question = input_text[question_start + len(question_marker):].strip()
            else:
                # If markers not found, return the whole input as question
                question = input_text.strip()
            return context, question

        # Extract context, question, and output for examples
        context1, question1 = extract_context_question(examples[0]['input'])
        output1 = examples[0]['output']

        context2, question2 = extract_context_question(examples[1]['input'])
        output2 = examples[1]['output']

        # Extract context and question for the current task
        context_task, question_task = extract_context_question(task['input'])

        # Format the examples with outputs
        example_1 = f"Context: {context1}\nQuestion: {question1}\nOutput: {output1}\n"
        example_2 = f"Context: {context2}\nQuestion: {question2}\nOutput: {output2}\n"

        # Instruction for the current task
        instruction = f"{definition}\n\nContext: {context_task}\nQuestion: {question_task}\n"

        # Create the prompt by injecting examples and instruction
        prompt = template.format(example_1=example_1, example_2=example_2, prompt=instruction)
        prompts.append(prompt)
    return prompts

def generate_initial_synthetic_data(base_model_name, data, synthetic_data_path, template, max_new_tokens=50):
    """
    Generates initial synthetic data using the base LLM and saves outputs in JSONL format.
    """
    # Load base model and tokenizer with BitsAndBytesConfig for quantization
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config
    )

    # Use a pipeline for generation
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Prepare prompts
    prompts = prepare_prompts(data, template)

    # Generate and save synthetic data
    synthetic_data = []
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating initial synthetic data"):
        try:
            response = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False  # Ensure only generated text is returned
            )
            generated_text = response[0]["generated_text"].strip()
            output_text = generated_text

            # Save the synthetic input and output
            synthetic_data.append({"input": prompt, "output": output_text})
        except Exception as e:
            print(f"Error generating response for prompt index {i}: {e}")

    # Save synthetic data to file
    with jsonlines.open(synthetic_data_path, mode="w") as writer:
        writer.write_all(synthetic_data)

    print(f"Synthetic data saved to {synthetic_data_path}")

def refine_synthetic_outputs(base_model_name, lora_weights_dir, synthetic_data_path, refined_data_path, max_new_tokens=50):
    """
    Refines the outputs of synthetic data using the latest LLM with LoRA weights and saves refined data.
    """
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA weights
    from peft import PeftModel, PeftConfig

    # Load PEFT configuration
    peft_config = PeftConfig.from_pretrained(lora_weights_dir)
    model = PeftModel.from_pretrained(base_model, lora_weights_dir)

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    # Use a pipeline for generation
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load synthetic data
    with jsonlines.open(synthetic_data_path, mode="r") as reader:
        synthetic_data = [item for item in reader]

    # Refine outputs
    refined_data = []
    for i, item in tqdm(enumerate(synthetic_data), total=len(synthetic_data), desc="Refining synthetic outputs"):
        synthetic_input = item['input']
        try:
            response = generator(
                synthetic_input,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            refined_output = response[0]["generated_text"].strip()
            # Save the refined input and output
            refined_data.append({"input": synthetic_input, "output": refined_output})
        except Exception as e:
            print(f"Error refining output for synthetic input index {i}: {e}")

    # Save refined data to file
    with jsonlines.open(refined_data_path, mode="w") as writer:
        writer.write_all(refined_data)

    print(f"Refined synthetic data saved to {refined_data_path}")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate and refine synthetic data using a language model with k-means sampling.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base HuggingFace model (θ(0)).")
    parser.add_argument("--lora_weights_dir", type=str, required=True, help="Directory containing LoRA weights for the latest model (θ(t-1)).")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the generated data files. Defaults to the directory of input_path.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("--sample_memory", type=int, default=200, help="Total number of samples to select.")
    parser.add_argument("--n_clusters", type=int, default=20, help="Number of clusters for k-means.")
    parser.add_argument("--text_field", type=str, default='input', help="Field in the data to generate embeddings from.")
    parser.add_argument("--template", type=str, default=llama_prompt, help="Prompt template.")
    parser.add_argument("--skip_initial_generation", action="store_true", help="Skip the initial synthetic data generation step.")
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_path) or '.'
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate default paths based on input_path and output_dir
    base_filename = os.path.splitext(os.path.basename(args.input_path))[0]
    embedding_path = os.path.join(args.output_dir, f"{base_filename}.embeddings.npy")
    synthetic_data_path = os.path.join(args.output_dir, f"{base_filename}.synthetic.jsonl")
    refined_data_path = os.path.join(args.output_dir, f"{base_filename}.refined.jsonl")
    output_path = os.path.join(args.output_dir, f"{base_filename}.final_sampled.jsonl")
    
    # Set random seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    
    # Load data
    with jsonlines.open(args.input_path, mode="r") as reader:
        data = [task for task in reader]

    # Step 1: Generate initial synthetic data using base LLM (θ(0))
    if not args.skip_initial_generation:
        generate_initial_synthetic_data(args.base_model_name, data, synthetic_data_path, args.template, args.max_new_tokens)
    else:
        print("Skipping initial synthetic data generation as per the argument.")

    # Step 2: Refine synthetic outputs using latest LLM (θ(t-1))
    refine_synthetic_outputs(args.base_model_name, args.lora_weights_dir, synthetic_data_path, refined_data_path, args.max_new_tokens)

    # Load refined synthetic data
    with jsonlines.open(refined_data_path, mode="r") as reader:
        refined_data = [item for item in reader]

    # Step 3: Generate embeddings for refined synthetic inputs
    embeddings = generate_embeddings(refined_data, text_field=args.text_field)

    # Verify that data and embeddings have the same length
    if len(refined_data) != embeddings.shape[0]:
        raise ValueError("The number of refined data points and embeddings must be the same.")

    # Perform clustering and sampling
    sampled_data = cluster_and_sample_data(refined_data, embeddings, sample_memory=args.sample_memory, n_clusters=args.n_clusters)

    # Save the final sampled data to output_path
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(sampled_data)

    print(f"Final sampled data saved to {output_path}")

if __name__ == "__main__":
    main()