import React from 'react';

interface ModelConfigurationProps {
  apiKeys: {
    anthropic: string;
    openai: string;
    huggingface: string;
  };
  onApiKeyChange: (provider: 'anthropic' | 'openai' | 'huggingface', value: string) => void;
}

const ModelConfiguration: React.FC<ModelConfigurationProps> = ({ 
  apiKeys, 
  onApiKeyChange 
}) => {
  return (
    <div className="model-config">
      <h2>API Keys Configuration</h2>
      <div className="api-keys">
        <div className="key-input">
          <label htmlFor="anthropic">Anthropic API Key:</label>
          <input 
            id="anthropic"
            type="password" 
            value={apiKeys.anthropic} 
            onChange={(e) => onApiKeyChange('anthropic', e.target.value)}
            placeholder="Enter Anthropic API key" 
          />
        </div>
        
        <div className="key-input">
          <label htmlFor="openai">OpenAI API Key:</label>
          <input 
            id="openai"
            type="password" 
            value={apiKeys.openai} 
            onChange={(e) => onApiKeyChange('openai', e.target.value)}
            placeholder="Enter OpenAI API key" 
          />
        </div>
        
        <div className="key-input">
          <label htmlFor="huggingface">HuggingFace API Key:</label>
          <input 
            id="huggingface"
            type="password" 
            value={apiKeys.huggingface} 
            onChange={(e) => onApiKeyChange('huggingface', e.target.value)}
            placeholder="Enter HuggingFace API key" 
          />
        </div>
      </div>
    </div>
  );
};

export default ModelConfiguration; 