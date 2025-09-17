import { Head, Html, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta charSet="utf-8" />
        <meta name="theme-color" content="#1e293b" />
        <meta name="description" content="Spectacular - Advanced AI reasoning with Bayesian Evidence Network and triple validation plots" />
        <meta name="keywords" content="AI, Bayesian network, machine learning, data visualization, reasoning, validation" />
        <meta name="author" content="Spectacular Framework" />
        
        {/* Open Graph / Facebook */}
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://spectacular.ai/" />
        <meta property="og:title" content="Spectacular - AI Reasoning Framework" />
        <meta property="og:description" content="Advanced AI chat interface with Bayesian Evidence Network, environmental sensor integration, and triple validation plots" />
        <meta property="og:image" content="/og-image.png" />

        {/* Twitter */}
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:url" content="https://spectacular.ai/" />
        <meta property="twitter:title" content="Spectacular - AI Reasoning Framework" />
        <meta property="twitter:description" content="Advanced AI chat interface with Bayesian Evidence Network and triple validation" />
        <meta property="twitter:image" content="/og-image.png" />

        {/* Favicon */}
        <link rel="icon" href="/favicon.ico" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/site.webmanifest" />
        
        {/* Preconnect to external domains for performance */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        
        {/* Font optimization */}
        <link 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap" 
          rel="stylesheet" 
        />
      </Head>
      <body>
        <Main />
        <NextScript />
        
        {/* Add custom scripts if needed */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Prevent flash of unstyled content
              document.documentElement.classList.add('loading');
              window.addEventListener('load', () => {
                document.documentElement.classList.remove('loading');
              });
            `,
          }}
        />
      </body>
    </Html>
  );
}
