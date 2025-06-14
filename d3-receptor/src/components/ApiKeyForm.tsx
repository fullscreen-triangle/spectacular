import React, { useState, useEffect } from 'react';

interface ApiKeys {
  anthropic: string;
  openai: string;
  huggingface: string;
}

interface ApiKeyFormProps {
  initialKeys: ApiKeys;
  onKeysChange: (provider: keyof ApiKeys, value: string) => void;
}

const ApiKeyForm: React.FC<ApiKeyFormProps> = ({ initialKeys, onKeysChange }) => {
  const [apiKeys, setApiKeys] = useState<ApiKeys>(initialKeys);

  // Update local state when initialKeys change
  useEffect(() => {
    setApiKeys(initialKeys);
  }, [initialKeys]);

  const handleChange = (provider: keyof ApiKeys, value: string) => {
    setApiKeys(prevKeys => ({
      ...prevKeys,
      [provider]: value
    }));
    onKeysChange(provider, value);
  };

  return (
    <div className="api-key-form">
      <h2>API Keys Configuration</h2>
      <div className="api-keys">
        <div className="key-input">
          <label htmlFor="anthropic">Anthropic API Key:</label>
          <input 
            id="anthropic"
            type="password" 
            value={apiKeys.anthropic} 
            onChange={(e) => handleChange('anthropic', e.target.value)}
            placeholder="Enter Anthropic API key" 
          />
        </div>
        
        <div className="key-input">
          <label htmlFor="openai">OpenAI API Key:</label>
          <input 
            id="openai"
            type="password" 
            value={apiKeys.openai} 
            onChange={(e) => handleChange('openai', e.target.value)}
            placeholder="Enter OpenAI API key" 
          />
        </div>
        
        <div className="key-input">
          <label htmlFor="huggingface">HuggingFace API Key:</label>
          <input 
            id="huggingface"
            type="password" 
            value={apiKeys.huggingface} 
            onChange={(e) => handleChange('huggingface', e.target.value)}
            placeholder="Enter HuggingFace API key" 
          />
        </div>
      </div>
    </div>
  );
};

export default ApiKeyForm; 