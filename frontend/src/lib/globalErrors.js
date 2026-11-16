/**
 * Global error handlers — catch unhandled promise rejections and
 * uncaught JavaScript errors, log them, and show a recovery UI.
 */
export function installGlobalErrorHandlers() {
  if (typeof window === 'undefined') return;

  const showBanner = (message) => {
    // Remove any existing banner
    const existing = document.getElementById('sisr-global-error-banner');
    if (existing) existing.remove();

    const banner = document.createElement('div');
    banner.id = 'sisr-global-error-banner';
    banner.className =
      'fixed top-0 left-0 right-0 z-50 p-4 bg-red-50 border border-red-200 text-red-900 text-sm font-medium';
    banner.innerHTML = `
      <div class="max-w-md mx-auto flex items-center gap-2">
        <svg class="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
        </svg>
        <span>${message}</span>
      </div>
      <button onclick="document.getElementById('sisr-global-error-banner').remove()" class="ml-auto text-xs underline hover:text-red-700">Dismiss</button>
    `;
    document.body.appendChild(banner);
  };

  window.addEventListener('unhandledrejection', (event) => {
    // eslint-disable-next-line no-console
    console.error('Unhandled promise rejection:', event.reason);
    showBanner(`App error: ${event.reason?.message || String(event.reason)}`);
    event.preventDefault();
  });

  window.addEventListener('error', (event) => {
    // eslint-disable-next-line no-console
    console.error('Uncaught JavaScript error:', event.error);
    showBanner(`App error: ${event.error?.message || event.message || 'Unknown error'}`);
  });
}
