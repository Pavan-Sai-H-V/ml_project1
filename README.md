# Llama Model Integration


## Creating and Running the Custom Model
After defining the rules and persona in the `Modelfile`, I created and ran the custom model named `david` using these commands:

```bash
ollama create david -f ./Modelfile
ollama run david
```

## Modelfile Purpose and Rules
The `Modelfile` is used to define the behavior and system instructions for the Llama model. In this project, the `Modelfile` sets the following:
- **Model Source:** llama3.1
- **Parameter:** temperature set to 1 for balanced creativity and determinism
- **System Prompt:** The model acts as a personal health trainer named David, answering as David the coach.

This configuration ensures that all responses from the Llama model follow the defined persona and rules, making it suitable for personal health coaching applications.
