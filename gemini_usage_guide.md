# Gemini RNA Inference Usage Guide

## Overview

The `inference_gemini.py` script provides RNA functional description generation using Google's Gemini models, similar to the ChatGPT inference script but using Google's Generative AI API.

## Prerequisites

1. **Install dependencies**:

   ```bash
   pip install google-generativeai python-dotenv
   ```

2. **Get Google API Key**:

   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Save it as `GOOGLE_API_KEY` in your environment or `.env` file

3. **Setup environment**:
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```bash
python inference_gemini.py
```

### Advanced Usage

```bash
# Use specific model and variant
python inference_gemini.py --model gemini-1.5-pro --variant name_and_sequence

# Process specific range of samples
python inference_gemini.py --start-index 0 --end-index 100

# Use custom API key
python inference_gemini.py --api-key YOUR_API_KEY
```

## Parameters

- `--api-key`: Google API key (default: from environment variable)
- `--model`: Gemini model to use (default: `gemini-1.5-pro`)
- `--variant`: Input variant:
  - `name_only`: Only provide RNA name
  - `name_and_sequence`: Provide both name and sequence
- `--start-index`: Starting index for evaluation (default: 4200)
- `--end-index`: Ending index for evaluation (default: 4210)

## Available Models

- `gemini-1.5-pro`: Latest Gemini Pro model (recommended)
- `gemini-1.5-flash`: Faster, lighter model
- `gemini-pro`: Previous generation model

## Output

The script generates:

- `results/gemini_{variant}_{model}.json`: Evaluation results with SimCSE scores
- Console output showing progress and sample results

## Comparison with ChatGPT

Both scripts (`inference_chatgpt.py` and `inference_gemini.py`) provide:

- Same evaluation metrics (SimCSE similarity)
- Same input variants (name_only, name_and_sequence)
- Same output format
- Same data processing pipeline

The main differences:

- **API**: Google Generative AI vs OpenAI
- **Models**: Gemini vs GPT models
- **Pricing**: Different cost structures
- **Performance**: Different strengths in biological text generation

## Example Output

```json
{
  "simcse_similarity": 0.7234,
  "num_samples": 10
}
```

## Troubleshooting

1. **API Key Issues**: Ensure your Google API key is valid and has sufficient quota
2. **Rate Limiting**: The script includes small delays to avoid rate limits
3. **Model Availability**: Some models may not be available in all regions
4. **Memory Issues**: For large datasets, consider processing in smaller batches

