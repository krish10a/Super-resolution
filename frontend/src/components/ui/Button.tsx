import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';

export interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'icon';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children?: React.ReactNode;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  children,
  icon,
  className = '',
  disabled,
  ...props
}) => {
  const baseClasses =
    'relative inline-flex items-center justify-center font-medium rounded-[var(--radius-control)] select-none transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]';

  const sizeClasses = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5 min-h-[32px]',
    md: 'text-sm px-4 py-2 gap-2 min-h-[40px]',
    lg: 'text-base px-5 py-2.5 gap-2.5 min-h-[48px]',
  }[size];

  const variantClasses = {
    primary:
      'bg-[var(--accent)] text-[#07090C] font-semibold hover:brightness-110 shadow-[0_0_15px_rgba(56,225,255,0.25)] hover:shadow-[0_0_20px_rgba(56,225,255,0.4)] disabled:opacity-50 disabled:cursor-not-allowed',
    secondary:
      'bg-[var(--raised)] text-[var(--text)] border border-[var(--border)] hover:bg-[var(--surface)] hover:border-[var(--muted)]/40 disabled:opacity-50',
    ghost:
      'bg-transparent text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--raised)]/60',
    icon: 'p-2 text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--raised)] min-w-[36px] min-h-[36px]',
  }[variant];

  return (
    <motion.button
      whileHover={{ scale: disabled || loading ? 1 : 1.02 }}
      whileTap={{ scale: disabled || loading ? 1 : 0.98 }}
      transition={{ duration: 0.15 }}
      disabled={disabled || loading}
      className={`${baseClasses} ${variant === 'icon' ? '' : sizeClasses} ${variantClasses} ${className}`}
      {...props}
    >
      {loading ? (
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-4 w-4 text-current" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span>Processing...</span>
        </span>
      ) : (
        <>
          {icon && <span className="shrink-0">{icon}</span>}
          {children}
        </>
      )}
    </motion.button>
  );
};

