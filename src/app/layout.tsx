import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    default: 'ProZoneSAM2 - 前列腺医学图像分割',
    template: '%s | ProZoneSAM2',
  },
  description:
    'ProZoneSAM2 - 基于 SAM2 的前列腺医学图像交互式分割工具，支持全腺体 (WG)、中央腺体 (CG) 和外周带 (PZ) 区域分割',
  keywords: [
    'ProZoneSAM2',
    'SAM2',
    '前列腺分割',
    '医学图像分割',
    'AI 分割',
    '前列腺',
    'MRI',
    '全腺体',
    '中央腺体',
    '外周带',
  ],
  authors: [{ name: 'Coze Code Team', url: 'https://code.coze.cn' }],
  generator: 'Coze Code',
  // icons: {
  //   icon: '',
  // },
  openGraph: {
    title: 'ProZoneSAM2 - 前列腺区域分割',
    description:
      '基于 SAM2 的前列腺医学图像交互式分割工具，支持全腺体、中央腺体和外周带区域分割',
    url: 'https://code.coze.cn',
    siteName: '扣子编程',
    locale: 'zh_CN',
    type: 'website',
    // images: [
    //   {
    //     url: '',
    //     width: 1200,
    //     height: 630,
    //     alt: '扣子编程 - 你的 AI 工程师',
    //   },
    // ],
  },
  // twitter: {
  //   card: 'summary_large_image',
  //   title: 'Coze Code | Your AI Engineer is Here',
  //   description:
  //     'Build and deploy full-stack applications through AI conversation. No env setup, just flow.',
  //   // images: [''],
  // },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
