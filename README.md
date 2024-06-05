## Development

Run using the following commands

```
python -m venv .venv
source ./.venv/bin/activate
pip install --upgrade -r requirements.txt
# Required, specifies where the LLM model specification is located
LLAMA_SPEC_PATH=models.json
# Optional, prints more debugging and output from provider
export GPTSCRIPT_DEBUG=true
./run.sh
```

```
gptscript --default-model="functionary from http://127.0.0.1:8000/v1" https://get.gptscript.ai/echo.gpt --input 'Hello, World!'
```

### `models.json`

```json
{
	"functionary": {
		"repo_id": "meetkai/functionary-small-v2.2-GGUF",
		"filename": "functionary-small-v2.2.q4_0.gguf",
		"chat_format": "functionary-v2",
		"n_ctx": 8096
	},
	"functionary-7b": {
		"repo_id": "meetkai/functionary-7b-v2.1-GGUF",
		"filename": "functionary-7b-v2.1.q4_0.gguf",
		"chat_format": "functionary-v2",
		"n_ctx": 8096
	}
}
```
