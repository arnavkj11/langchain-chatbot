"use client";

import ReactMarkdown from "react-markdown";
import { memo } from "react";
import remarkGfm from "remark-gfm";
import { useTheme } from "./ThemeProvider";

const MarkdownRenderer = memo(({ content }: { content: string }) => {
  const { theme } = useTheme();
  const parsedContent = content.replace(/\\n/g, "\n"); // Parse the escape sequences to convert \n to actual linebreaks

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, inline, className, children, ...props }: any) {
          return !inline ? (
            <pre
              className={`bg-gray-900 dark:bg-gray-800 text-gray-100 dark:text-gray-200 p-4 rounded-lg overflow-x-auto text-sm font-mono border border-gray-700 dark:border-gray-600 ${className}`}
              {...props}
            >
              <code>{children}</code>
            </pre>
          ) : (
            <code
              className={`${className} bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 px-1 py-0.5 rounded text-sm font-mono`}
              {...props}
            >
              {children}
            </code>
          );
        },
        p: ({ children, ...props }: any) => (
          <p
            className="text-gray-700 dark:text-gray-300 leading-relaxed mb-3"
            {...props}
          >
            {children}
          </p>
        ),
        h1: ({ children, ...props }: any) => (
          <h1
            className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4"
            {...props}
          >
            {children}
          </h1>
        ),
        h2: ({ children, ...props }: any) => (
          <h2
            className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-3"
            {...props}
          >
            {children}
          </h2>
        ),
        h3: ({ children, ...props }: any) => (
          <h3
            className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2"
            {...props}
          >
            {children}
          </h3>
        ),
        ul: ({ children, ...props }: any) => (
          <ul
            className="list-disc list-inside text-gray-700 dark:text-gray-300 mb-3 space-y-1"
            {...props}
          >
            {children}
          </ul>
        ),
        ol: ({ children, ...props }: any) => (
          <ol
            className="list-decimal list-inside text-gray-700 dark:text-gray-300 mb-3 space-y-1"
            {...props}
          >
            {children}
          </ol>
        ),
        li: ({ children, ...props }: any) => (
          <li className="text-gray-700 dark:text-gray-300" {...props}>
            {children}
          </li>
        ),
        blockquote: ({ children, ...props }: any) => (
          <blockquote
            className="border-l-4 border-blue-500 dark:border-cyan-400 bg-blue-50 dark:bg-gray-700 text-blue-900 dark:text-blue-100 p-4 my-4 rounded-r-lg"
            {...props}
          >
            {children}
          </blockquote>
        ),
        a: ({ children, href, ...props }: any) => (
          <a
            href={href}
            className="text-blue-600 dark:text-cyan-400 hover:text-blue-800 dark:hover:text-cyan-300 underline transition-colors"
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        ),
        strong: ({ children, ...props }: any) => (
          <strong
            className="font-semibold text-gray-900 dark:text-gray-100"
            {...props}
          >
            {children}
          </strong>
        ),
        em: ({ children, ...props }: any) => (
          <em className="italic text-gray-800 dark:text-gray-200" {...props}>
            {children}
          </em>
        ),
      }}
    >
      {parsedContent}
    </ReactMarkdown>
  );
});

export default MarkdownRenderer;
