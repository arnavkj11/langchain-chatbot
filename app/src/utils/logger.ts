/**
 * Frontend Logging Utility
 * Provides consistent logging across all frontend components
 */

interface LogData {
  [key: string]: any;
}

export class Logger {
  private component: string;
  private static isDevelopment = process.env.NODE_ENV === 'development';
  
  constructor(component: string) {
    this.component = component;
  }

  private formatMessage(level: string, message: string): string {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    const emoji = this.getEmoji(level);
    return `${emoji} [${timestamp}] [${this.component}] ${message}`;
  }

  private getEmoji(level: string): string {
    switch (level.toLowerCase()) {
      case 'info': return '🔵';
      case 'success': return '🟢';
      case 'warn': case 'warning': return '🟡';
      case 'error': return '🔴';
      case 'debug': return '🟣';
      default: return '⚪';
    }
  }

  info(message: string, data?: LogData): void {
    if (!Logger.isDevelopment) return;
    console.log(this.formatMessage('info', message), data || '');
  }

  success(message: string, data?: LogData): void {
    if (!Logger.isDevelopment) return;
    console.log(this.formatMessage('success', message), data || '');
  }

  warn(message: string, data?: LogData): void {
    if (!Logger.isDevelopment) return;
    console.warn(this.formatMessage('warn', message), data || '');
  }

  error(message: string, error?: any): void {
    console.error(this.formatMessage('error', message), error || '');
  }

  debug(message: string, data?: LogData): void {
    if (!Logger.isDevelopment) return;
    console.debug(this.formatMessage('debug', message), data || '');
  }

  // Performance timing utilities
  startTimer(label: string): void {
    if (!Logger.isDevelopment) return;
    console.time(`⏱️ [${this.component}] ${label}`);
  }

  endTimer(label: string): void {
    if (!Logger.isDevelopment) return;
    console.timeEnd(`⏱️ [${this.component}] ${label}`);
  }

  // Group logging for related operations
  group(label: string): void {
    if (!Logger.isDevelopment) return;
    console.group(`📁 [${this.component}] ${label}`);
  }

  groupEnd(): void {
    if (!Logger.isDevelopment) return;
    console.groupEnd();
  }
}

// Factory function for creating loggers
export const createLogger = (component: string): Logger => {
  return new Logger(component);
};

// Global application logger
export const appLogger = createLogger('App');
