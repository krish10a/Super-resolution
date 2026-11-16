import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const projectRoot = path.resolve(__dirname, '..', '..');
const evalSourceDir = path.join(projectRoot, 'outputs', 'evaluation');
const targetDataDir = path.resolve(__dirname, '..', 'public', 'data');

if (!fs.existsSync(targetDataDir)) {
  fs.mkdirSync(targetDataDir, { recursive: true });
}

console.log('[sync-data] Syncing evaluation outputs to frontend/public/data/...');

if (fs.existsSync(evalSourceDir)) {
  const files = fs.readdirSync(evalSourceDir);
  let count = 0;
  for (const file of files) {
    if (file.endsWith('.json') || file.endsWith('.log') || file.endsWith('.csv')) {
      const srcPath = path.join(evalSourceDir, file);
      const destPath = path.join(targetDataDir, file);
      fs.copyFileSync(srcPath, destPath);
      count++;
    }
  }
  console.log(`[sync-data] Successfully synced ${count} data files.`);
} else {
  console.warn(`[sync-data] Evaluation source dir not found: ${evalSourceDir}`);
}

