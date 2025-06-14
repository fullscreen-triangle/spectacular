export interface GenerationOptions {
  model?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  seed?: number;
}

export interface GenerationResponse {
  model: string;
  response: string;
  totalDuration?: number;
  promptEvalCount?: number;
  evalCount?: number;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ModelInfo {
  name: string;
  modelType: string;
  parameters?: number;
  quantizationType?: string;
  created?: number;
  modified?: number;
}

export class OllamaService {
  private baseUrl: string;
  private defaultModel: string;
  
  constructor(baseUrl: string = 'http://localhost:11434', defaultModel: string = 'd3-neuro') {
    this.baseUrl = baseUrl;
    this.defaultModel = defaultModel;
  }
  
  /**
   * Generate a response using the Ollama model
   */
  async generate(prompt: string, options?: GenerationOptions): Promise<GenerationResponse> {
    const modelName = options?.model || this.defaultModel;
    
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: modelName,
        prompt,
        temperature: options?.temperature,
        max_tokens: options?.max_tokens,
        top_p: options?.top_p,
        top_k: options?.top_k,
        stop: options?.stop,
        seed: options?.seed
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to generate response: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      model: modelName,
      response: data.response,
      totalDuration: data.total_duration,
      promptEvalCount: data.prompt_eval_count,
      evalCount: data.eval_count,
      usage: {
        prompt_tokens: data.prompt_tokens,
        completion_tokens: data.completion_tokens,
        total_tokens: data.prompt_tokens + data.completion_tokens
      }
    };
  }
  
  /**
   * Stream a response from the model
   */
  async streamGenerate(
    prompt: string, 
    callback: (chunk: string) => void, 
    options?: GenerationOptions
  ): Promise<void> {
    const modelName = options?.model || this.defaultModel;
    
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/x-ndjson'
      },
      body: JSON.stringify({
        model: modelName,
        prompt,
        stream: true,
        temperature: options?.temperature,
        max_tokens: options?.max_tokens,
        top_p: options?.top_p,
        top_k: options?.top_k,
        stop: options?.stop,
        seed: options?.seed
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to stream response: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Response body is empty');

    const decoder = new TextDecoder();
    let done = false;

    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;
      
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            callback(data.response);
          }
        } catch (e) {
          console.error('Error parsing streaming response:', e);
        }
      }
    }
  }
  
  /**
   * Get model information
   */
  async getModelInfo(modelName: string = this.defaultModel): Promise<ModelInfo> {
    const response = await fetch(`${this.baseUrl}/api/show`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: modelName
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      name: data.name,
      modelType: data.model_type || 'unknown',
      parameters: data.parameters,
      quantizationType: data.quantization_type,
      created: data.created_at,
      modified: data.modified_at
    };
  }
  
  /**
   * List available models
   */
  async listModels(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/tags`, {
      method: 'GET'
    });

    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }

    const data = await response.json();
    return data.models?.map((model: { name: string }) => model.name) || [];
  }
} 