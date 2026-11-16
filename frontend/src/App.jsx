/**
 * App root — renders the Layout shell around the hash-router pages.
 * Routes are defined in AppRouter.jsx (/, /studio, /compare, /info).
 */
import Layout from './components/Layout.jsx';
import AppRouter from './AppRouter.jsx';

export default function App() {
  return (
    <Layout>
      <AppRouter />
    </Layout>
  );
}
