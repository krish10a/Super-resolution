/**
 * React Error Boundary — catches rendering errors anywhere in the component
 * tree and shows a recovery UI instead of a blank white screen.
 */
import React, { Component } from 'react';

export class ErrorBoundary extends Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 text-center bg-red-50 border border-red-200 rounded-2xl">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Something went wrong</h3>
          <p className="text-sm text-red-700 mb-4">{this.state.error?.message || 'Unknown error'}</p>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
            className="text-xs underline underline-offset-4 hover:text-red-700"
          >
            Refresh page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
