/**
 * Tiny class-name merger (tailwind-merge / clsx substitute).
 * Joins class strings and filters out falsy values.
 */
export function cn(...classes) {
  return classes.filter(Boolean).join(' ');
}
