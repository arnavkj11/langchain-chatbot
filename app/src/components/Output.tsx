import MarkdownRenderer from "@/components/MarkdownRenderer";
import { Step, type ChatOutput } from "@/types";
import { useEffect, useState } from "react";

// Logging utility for Output component
const log = {
  info: (message: string, data?: any) => {
    console.log(`🟦 [Output] ${message}`, data ? data : '');
  },
  debug: (message: string, data?: any) => {
    console.debug(`🟨 [Output] ${message}`, data ? data : '');
  }
};

const Output = ({ output }: { output: ChatOutput }) => {
  const detailsHidden = !!output.result?.answer;
  
  useEffect(() => {
    log.info('Output component rendered', {
      question: output.question.substring(0, 50) + (output.question.length > 50 ? '...' : ''),
      stepsCount: output.steps.length,
      hasAnswer: !!output.result?.answer,
      toolsUsed: output.result?.tools_used || []
    });
  }, [output]);
  
  return (
    <div className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-sm rounded-2xl border border-white/50 dark:border-gray-600/50 shadow-lg p-6 hover:shadow-xl transition-all duration-300 message-appear">
      {/* Question */}
      <div className="mb-6">
        <div className="flex items-start space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-500 dark:from-emerald-400 dark:to-cyan-400 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
            <svg
              className="w-4 h-4 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
              />
            </svg>
          </div>
          <div className="flex-1">
            <p className="text-lg font-semibold text-gray-800 dark:text-gray-100 leading-relaxed">
              {output.question}
            </p>
          </div>
        </div>
      </div>

      {/* Steps */}
      {output.steps.length > 0 && (
        <GenerationSteps steps={output.steps} done={detailsHidden} />
      )}

      {/* Answer */}
      {output.result?.answer && (
        <div className="mt-6 fade-in">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 dark:from-violet-400 dark:to-fuchsia-400 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <svg
                className="w-4 h-4 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
            <div className="flex-1 bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-700 dark:to-gray-600 rounded-xl p-4 border border-gray-200 dark:border-gray-600">
              <div
                className="prose prose-gray dark:prose-invert max-w-none"
                style={{ overflowWrap: "anywhere" }}
              >
                <MarkdownRenderer content={output.result.answer} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tools */}
      {output.result?.tools_used?.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
            <span className="font-medium">Tools used:</span>
            <div className="flex flex-wrap gap-1">
              {output.result.tools_used.map((tool, i) => (
                <span
                  key={i}
                  className="px-2 py-1 bg-gradient-to-r from-blue-500 to-purple-500 dark:from-cyan-400 dark:to-purple-400 text-white text-xs rounded-full font-medium"
                >
                  {tool}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const GenerationSteps = ({ steps, done }: { steps: Step[]; done: boolean }) => {
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    log.debug('GenerationSteps updated', {
      stepsCount: steps.length,
      done,
      stepNames: steps.map(s => s.name)
    });
    if (done) {
      log.info('Generation completed - hiding steps');
      setHidden(true);
    }
  }, [done, steps]);

  const toggleSteps = () => {
    setHidden(!hidden);
    log.debug(`Steps visibility toggled: ${hidden ? 'showing' : 'hiding'}`);
  };

  return (
    <div className="bg-white/50 backdrop-blur-sm border border-gray-200 rounded-xl mt-4 p-4 shadow-sm">
      <button
        className="w-full text-left flex items-center justify-between hover:bg-gray-50 rounded-lg p-2 transition-colors"
        onClick={toggleSteps}
      >
        <div className="flex items-center space-x-2">
          <div
            className={`w-3 h-3 rounded-full transition-colors ${
              !done
                ? "animate-pulse bg-gradient-to-r from-blue-500 to-purple-500"
                : "bg-gray-400"
            }`}
          />
          <span className="font-medium text-gray-800">
            {done ? "Processing Steps" : "Processing..."}
          </span>
        </div>
        {hidden ? <ChevronDown /> : <ChevronUp />}
      </button>

      {!hidden && (
        <div className="mt-4 space-y-3">
          {steps.map((step, j) => (
            <div
              key={j}
              className="bg-white rounded-lg p-3 border border-gray-100 shadow-sm"
            >
              <h4 className="font-medium text-gray-800 mb-2 capitalize">
                {step.name.replace(/_/g, " ")}
              </h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(step.result).map(([key, value]) => (
                  <span
                    key={key}
                    className="inline-flex items-center px-2 py-1 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-700 text-xs rounded-md font-medium"
                  >
                    <span className="text-gray-500">{key}:</span>
                    <span className="ml-1">{value}</span>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const ChevronDown = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="text-gray-500 transition-transform duration-200"
  >
    <path d="m6 9 6 6 6-6" />
  </svg>
);

const ChevronUp = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="text-gray-500 transition-transform duration-200"
  >
    <path d="m18 15-6-6-6 6" />
  </svg>
);

export default Output;
