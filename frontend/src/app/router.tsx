import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppLayout } from './AppLayout';
import { Overview } from '../pages/Overview.tsx';
import { Studio } from '../pages/Studio.tsx';
import { Benchmarks } from '../pages/Benchmarks.tsx';
import { Models } from '../pages/Models.tsx';
import { Method } from '../pages/Method.tsx';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <Overview />,
      },
      {
        path: 'studio',
        element: <Studio />,
      },
      {
        path: 'benchmarks',
        element: <Benchmarks />,
      },
      {
        path: 'models',
        element: <Models />,
      },
      {
        path: 'method',
        element: <Method />,
      },
      {
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);
