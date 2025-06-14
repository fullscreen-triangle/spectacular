import React from 'react';
import { ModelRegistry } from '../services';

interface ModelInfoProps {
  modelRegistry: ModelRegistry;
}

const ModelInfo: React.FC<ModelInfoProps> = ({ modelRegistry }) => {
  const [modelDetails, setModelDetails] = React.useState<string[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);

  React.useEffect(() => {
    async function loadModelDetails() {
      try {
        const models = modelRegistry.getAllModels();
        
        const modelInfoMessages = await Promise.all(
          models.map(async model => {
            const info = await model.service.getModelInfo();
            return `${info.name} (${info.provider}):\n  - Tasks: ${model.tasks.join(', ')}\n  - Capabilities: ${info.capabilities.join(', ')}`;
          })
        );
        
        setModelDetails(modelInfoMessages);
      } catch (err) {
        console.error('Error loading model details:', err);
      } finally {
        setIsLoading(false);
      }
    }
    
    loadModelDetails();
  }, [modelRegistry]);

  if (isLoading) {
    return <p className="loading">Loading model details...</p>;
  }

  if (modelDetails.length === 0) {
    return <p>No models registered</p>;
  }

  return (
    <div className="model-info">
      <h3>Registered Models</h3>
      <pre className="model-details">
        {modelDetails.join('\n\n')}
      </pre>
    </div>
  );
};

export default ModelInfo; 