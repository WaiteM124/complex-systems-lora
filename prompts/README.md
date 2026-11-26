# Prompts

Versioned prompt files live here so they are reviewed in pull requests rather than buried in code.

## Naming

Use `{name}.v{N}.txt`, for example `chat_system.v1.txt`. When you change a prompt's behaviour, add a new version file and point the application at it. Keep the previous version until you are sure nothing needs it.

## Rules

- Do not put secrets, keys, or real account IDs in a prompt file.
- Treat user messages and retrieved text as untrusted data, not as instructions. Say so in the system prompt.
- Load prompts with `app.prompts.load_prompt(name, version)`, not by hardcoding the text in Python.
