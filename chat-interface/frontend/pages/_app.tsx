import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import { useEffect } from 'react';

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    // Disable right-click context menu in production
    if (process.env.NODE_ENV === 'production') {
      document.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    // Add keyboard shortcuts
    const handleKeyboard = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K to focus search (if implemented)
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        // Focus on chat input
        const chatInput = document.querySelector('textarea[placeholder*="Ask me anything"]') as HTMLTextAreaElement;
        if (chatInput) {
          chatInput.focus();
        }
      }
    };
    
    document.addEventListener('keydown', handleKeyboard);
    
    // Cleanup
    return () => {
      document.removeEventListener('contextmenu', (e) => e.preventDefault());
      document.removeEventListener('keydown', handleKeyboard);
    };
  }, []);

  return <Component {...pageProps} />;
}
