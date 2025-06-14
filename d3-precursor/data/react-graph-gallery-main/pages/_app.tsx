import 'tailwindcss/tailwind.css';
import '../style/style.css';
import type { AppProps } from 'next/app';
import Script from 'next/script';
import '../style/prism.css';
import { bitterFontClass, robotoFontClass } from 'util/fonts';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Script
        strategy="lazyOnload"
        src={`https://www.googletagmanager.com/gtag/js?id=G-0Q8EEVB6R4`}
      />
      <Script id="ga-analytics">
        {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());

            gtag('config', 'G-0Q8EEVB6R4');
          `}
      </Script>
      <main className={bitterFontClass + ' ' + robotoFontClass}>
        <Component {...pageProps} />
      </main>
    </>
  );
}

export default MyApp;
