import { useState, useEffect, useMemo, useCallback, lazy, Suspense } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { ModelRegistry, OllamaService } from './services'

// Lazy load components for code splitting
const ModelInfo = lazy(() => import('./components/ModelInfo'));
const ApiKeyForm = lazy(() => import('./components/ApiKeyForm'));

// Add performance monitoring
const reportPerformance = () => {
  if (window.performance && 'getEntriesByType' in window.performance) {
    const perfEntries = window.performance.getEntriesByType('navigation');
    if (perfEntries.length > 0) {
      const navigationEntry = perfEntries[0] as PerformanceNavigationTiming;
      console.log('App Performance Metrics:');
      console.log(`- DOM Content Loaded: ${navigationEntry.domContentLoadedEventEnd - navigationEntry.startTime}ms`);
      console.log(`- Load Event: ${navigationEntry.loadEventEnd - navigationEntry.startTime}ms`);
      console.log(`- First Paint: ${window.performance.getEntriesByName('first-paint')[0]?.startTime || 'N/A'}ms`);
    }
  }
};

function App() {
  const [count, setCount] = useState(0)
  const [modelInfo, setModelInfo] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiKeys, setApiKeys] = useState({
    anthropic: localStorage.getItem('anthropic_api_key') || '',
    openai: localStorage.getItem('openai_api_key') || '',
    huggingface: localStorage.getItem('huggingface_api_key') || '',
  })
  const [showModelDetails, setShowModelDetails] = useState(false)
  
  // Use useMemo to create modelRegistry to prevent recreation on each render
  const modelRegistry = useMemo(() => new ModelRegistry({
    apiKeys: {
      anthropic: apiKeys.anthropic || undefined,
      openai: apiKeys.openai || undefined,
      huggingface: apiKeys.huggingface || undefined,
    },
    enableCache: true, // Enable response caching
  }), [apiKeys.anthropic, apiKeys.openai, apiKeys.huggingface])
  
  // Use useCallback for event handlers
  const handleApiKeyChange = useCallback((provider: 'anthropic' | 'openai' | 'huggingface', value: string) => {
    setApiKeys(prevKeys => {
      const newKeys = { ...prevKeys, [provider]: value };
      localStorage.setItem(`${provider}_api_key`, value);
      
      // Update model registry with new keys
      modelRegistry.updateApiKeys({
        [provider]: value || undefined
      });
      
      return newKeys;
    });
  }, [modelRegistry]);
  
  const incrementCount = useCallback(() => {
    setCount(prevCount => prevCount + 1);
  }, []);
  
  const toggleModelDetails = useCallback(() => {
    setShowModelDetails(prev => !prev);
  }, []);
  
  // Initialize models when API keys change
  useEffect(() => {
    let isMounted = true;
    
    async function initializeModels() {
      try {
        if (!isMounted) return;
        
        setIsLoading(true);
        setError(null);
        
        // If no API keys are configured, try to connect to local Ollama as fallback
        if (!apiKeys.anthropic && !apiKeys.openai && !apiKeys.huggingface) {
          try {
            const ollamaService = new OllamaService();
            const models = await ollamaService.listModels();
            if (isMounted) {
              setModelInfo(`Using local Ollama models: ${models.join(', ')}`);
            }
          } catch (err) {
            if (isMounted) {
              setError(`No API keys configured and failed to connect to Ollama: ${err instanceof Error ? err.message : String(err)}`);
            }
          }
          if (isMounted) {
            setIsLoading(false);
          }
          return;
        }
        
        // Initialize with recommended models
        await modelRegistry.initializeRecommendedModels();
        
        if (!isMounted) return;
        
        // Get model descriptions and update UI
        const modelInfoMessages = await Promise.all(
          modelRegistry.getAllModels().map(async model => {
            const info = await model.service.getModelInfo();
            return `${info.name} (${model.tasks.join(', ')})`;
          })
        );
        
        if (isMounted) {
          setModelInfo(`Using specialized models:\n${modelInfoMessages.join('\n')}`);
        }
      } catch (err) {
        if (isMounted) {
          setError(`Failed to initialize models: ${err instanceof Error ? err.message : String(err)}`);
          console.error(err);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }
    
    initializeModels();
    
    // Cleanup function to prevent state updates after unmount
    return () => {
      isMounted = false;
    };
  }, [apiKeys.anthropic, apiKeys.openai, apiKeys.huggingface, modelRegistry]);

  // Report performance on first render
  useEffect(() => {
    // Wait until the page has fully loaded
    window.addEventListener('load', reportPerformance);
    return () => window.removeEventListener('load', reportPerformance);
  }, []);

  // Memoize the UI elements that don't change often
  const headerSection = useMemo(() => (
    <>
      <div>
        <a href="https://vite.dev" target="_blank" rel="noopener noreferrer">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank" rel="noopener noreferrer">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>D3-Neuro with Specialized Models</h1>
    </>
  ), []);

  return (
    <>
      {headerSection}
      
      <div className="card">
        <Suspense fallback={<div>Loading API key form...</div>}>
          <ApiKeyForm 
            initialKeys={apiKeys} 
            onKeysChange={handleApiKeyChange}
          />
        </Suspense>
        
        <div className="counter">
          <button onClick={incrementCount}>
            count is {count}
          </button>
        </div>
        
        {isLoading && <p className="loading">Loading model information...</p>}
        {error && <p className="error">Error: {error}</p>}
        {modelInfo && <pre className="models">{modelInfo}</pre>}
        
        <button onClick={toggleModelDetails}>
          {showModelDetails ? 'Hide' : 'Show'} Detailed Model Info
        </button>
        
        {showModelDetails && (
          <Suspense fallback={<div>Loading model details...</div>}>
            <ModelInfo modelRegistry={modelRegistry} />
          </Suspense>
        )}
      </div>
      
      <p className="read-the-docs">
        D3-Neuro uses specialized models for different visualization tasks
      </p>
    </>
  )
}

export default App
