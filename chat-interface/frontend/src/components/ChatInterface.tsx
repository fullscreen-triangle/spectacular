import { SpectacularAPI } from '@/lib/api';
import { BayesianNetworkResults, ChatMessage, ChatRequest, ChatResponse, ValidationDetails } from '@/types/api';
import { AlertCircle, Brain, CheckCircle2, Loader2, Send, TrendingUp, Zap } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { v4 as uuidv4 } from 'uuid';
import PlotDisplay from './PlotDisplay';

interface ChatInterfaceProps {
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ className = '' }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputMessage]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isLoading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: uuidv4(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const messageToSend = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    try {
      const request: ChatRequest = {
        message: messageToSend,
        conversation_id: currentConversationId || undefined,
        context: {
          timestamp: new Date().toISOString(),
          interface: 'web_frontend'
        }
      };

      const response: ChatResponse = await SpectacularAPI.sendChatMessage(request);

      // Set conversation ID if not set
      if (!currentConversationId) {
        setCurrentConversationId(response.conversation_id);
      }

      const assistantMessage: ChatMessage = {
        id: uuidv4(),
        type: 'assistant',
        content: response.response_text,
        timestamp: new Date(),
        plots: response.plots,
        validation_details: response.validation_details,
        bayesian_results: response.validation_details.bayesian_intelligence ? 
          extractBayesianResults(response) : undefined
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Show success toast
      toast.success(
        `Query processed successfully! Coherence: ${(response.coherence_score * 100).toFixed(1)}%`,
        { duration: 4000 }
      );

    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        type: 'assistant',
        content: `I apologize, but I encountered an error while processing your request. Please try again.\n\nError: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
      toast.error('Failed to process message. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const extractBayesianResults = (response: ChatResponse): BayesianNetworkResults | undefined => {
    // Extract Bayesian network results from validation details if available
    const details = response.validation_details;
    if (!details.bayesian_intelligence) return undefined;

    return {
      nodes_converged: details.pipeline_stages_completed || 0,
      total_nodes: details.fuzzy_logic_nodes || 8,
      recursive_loops_executed: {},
      external_validations_passed: details.external_validation ? 1 : 0,
      network_coherence: response.coherence_score,
      coherence_trajectory: details.confidence_progression || [],
      embedding_paths: {}
    };
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e as any);
    }
  };

  const renderValidationSummary = (details: ValidationDetails, bayesianResults?: BayesianNetworkResults) => {
    return (
      <div className="mt-4 p-4 glass-effect rounded-lg">
        <h4 className="text-white font-medium mb-3 flex items-center">
          <Brain className="w-4 h-4 mr-2 text-blue-400" />
          Processing Summary
        </h4>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Pipeline Stages:</span>
            <span className="text-blue-400">{details.pipeline_stages_completed}/8</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Processing Time:</span>
            <span className="text-yellow-400">{details.stage_timings ? 
              Object.values(details.stage_timings).reduce((a, b) => a + b, 0).toFixed(2) + 's' : 'N/A'}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Environmental:</span>
            <span className={`flex items-center ${details.environmental_integration ? 'text-green-400' : 'text-red-400'}`}>
              {details.environmental_integration ? (
                <CheckCircle2 className="w-3 h-3 mr-1" />
              ) : (
                <AlertCircle className="w-3 h-3 mr-1" />
              )}
              {details.environmental_integration ? 'Yes' : 'No'}
            </span>
          </div>
          
          {details.bayesian_intelligence && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Bayesian Nodes:</span>
                <span className="text-purple-400">{details.fuzzy_logic_nodes || 0}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Dynamic Routing:</span>
                <span className={`flex items-center ${details.dynamic_routing ? 'text-green-400' : 'text-gray-400'}`}>
                  {details.dynamic_routing ? (
                    <CheckCircle2 className="w-3 h-3 mr-1" />
                  ) : (
                    <AlertCircle className="w-3 h-3 mr-1" />
                  )}
                  {details.dynamic_routing ? 'Yes' : 'No'}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Recursion:</span>
                <span className={`flex items-center ${details.recursive_processing ? 'text-orange-400' : 'text-gray-400'}`}>
                  {details.recursive_processing ? (
                    <TrendingUp className="w-3 h-3 mr-1" />
                  ) : (
                    <AlertCircle className="w-3 h-3 mr-1" />
                  )}
                  {details.recursive_processing ? 'Used' : 'Not used'}
                </span>
              </div>
            </>
          )}
        </div>
        
        {bayesianResults && (
          <div className="mt-3 pt-3 border-t border-white/10">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-300">Network Coherence:</span>
              <div className="flex items-center">
                <div className="w-16 h-2 bg-gray-700 rounded-full mr-2">
                  <div 
                    className="h-2 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
                    style={{ width: `${bayesianResults.network_coherence * 100}%` }}
                  ></div>
                </div>
                <span className="text-white font-mono text-xs">
                  {(bayesianResults.network_coherence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderMessage = (message: ChatMessage) => {
    if (message.type === 'user') {
      return (
        <div key={message.id} className="flex justify-end mb-6">
          <div className="max-w-3xl">
            <div className="glass-effect rounded-xl px-4 py-3 text-white">
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
            <div className="text-xs text-gray-400 mt-1 text-right">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div key={message.id} className="flex justify-start mb-8">
        <div className="max-w-full w-full">
          {/* Assistant Response Text */}
          <div className="glass-effect rounded-xl px-6 py-4 mb-4">
            <div className="flex items-center mb-3">
              <Brain className="w-5 h-5 text-blue-400 mr-2" />
              <span className="text-white font-medium">Spectacular AI</span>
              {message.validation_details?.bayesian_intelligence && (
                <span className="ml-2 text-xs px-2 py-1 bg-purple-600/30 text-purple-300 rounded-full">
                  Bayesian Network
                </span>
              )}
            </div>
            <div className="chat-message">
              <p className="text-gray-100 leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            </div>
          </div>

          {/* Validation Summary */}
          {message.validation_details && renderValidationSummary(
            message.validation_details, 
            message.bayesian_results
          )}

          {/* Triple Validation Plots */}
          {message.plots && (
            <div className="mt-6 space-y-6">
              <div className="text-white font-semibold text-lg mb-4 flex items-center">
                <Zap className="w-5 h-5 text-yellow-400 mr-2" />
                Triple Validation Analysis
              </div>
              
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <PlotDisplay 
                  plotData={message.plots.ridiculous}
                  type="ridiculous"
                />
                <PlotDisplay 
                  plotData={message.plots.intent}
                  type="intent"
                />
                <PlotDisplay 
                  plotData={message.plots.reasoning}
                  type="reasoning"
                />
              </div>
            </div>
          )}

          <div className="text-xs text-gray-400 mt-2">
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-300 mt-12">
            <Brain className="w-16 h-16 mx-auto mb-4 text-blue-400 opacity-50" />
            <h3 className="text-xl font-semibold mb-2">Welcome to Spectacular</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Ask me anything and I'll analyze it through our Bayesian Evidence Network 
              with triple validation plots and environmental sensor integration.
            </p>
            <div className="mt-6 text-sm text-gray-500">
              <p>✨ Try asking: "How does Newton's second law work?"</p>
              <p>🧠 Or: "Explain quantum entanglement with visual proof"</p>
            </div>
          </div>
        )}
        
        {messages.map(renderMessage)}
        
        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex justify-start mb-6">
            <div className="glass-effect rounded-xl px-6 py-4 flex items-center space-x-3">
              <div className="processing-bayesian">
                <Brain className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <div className="text-white font-medium">Processing through Bayesian Network...</div>
                <div className="text-gray-300 text-sm mt-1">
                  Analyzing evidence, routing through nodes, validating externally...
                </div>
              </div>
              <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="border-t border-white/10 p-6">
        <form onSubmit={handleSendMessage} className="flex space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything... I'll analyze it through the Bayesian Evidence Network"
              className="chat-input w-full resize-none overflow-hidden"
              rows={1}
              disabled={isLoading}
              style={{ maxHeight: '120px' }}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading || !inputMessage.trim()}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            <span className="hidden sm:inline">
              {isLoading ? 'Processing' : 'Send'}
            </span>
          </button>
        </form>
        
        <div className="mt-2 text-xs text-gray-400 text-center">
          Powered by Spectacular Bayesian Evidence Network • Environmental Sensor Integration • Triple Validation
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
