import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';

export interface CardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
  variant?: 'surface' | 'raised' | 'base';
  interactive?: boolean;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'surface',
  interactive = false,
  className = '',
  ...props
}) => {
  const bgClasses = {
    surface: 'bg-[var(--surface)] border-[var(--border)]',
    raised: 'bg-[var(--raised)] border-[var(--border)]',
    base: 'bg-[var(--bg-base)] border-[var(--border)]',
  }[variant];

  return (
    <motion.div
      whileHover={interactive ? { y: -2, borderColor: 'rgba(56, 225, 255, 0.3)' } : undefined}
      transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
      className={`rounded-[var(--radius-card)] border p-5 ${bgClasses} ${
        interactive ? 'cursor-pointer transition-colors' : ''
      } ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  );
};

