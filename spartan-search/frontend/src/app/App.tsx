import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";
import { BrowserRouter, Outlet, Route, Routes, useLocation } from "react-router-dom";

import { Footer, NavBar } from "../components/NavBar";
import { AboutPage } from "../pages/About";
import { BenchmarksPage } from "../pages/Benchmarks";
import { ComparePage } from "../pages/Compare";
import { HomePage } from "../pages/Home";
import { LibraryPage } from "../pages/Library";
import { NotFoundPage } from "../pages/NotFound";
import { SearchPage } from "../pages/Search";
import { UploadPage } from "../pages/Upload";
import { VideoDetailPage } from "../pages/VideoDetail";

export function makeQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
  });
}

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => window.scrollTo(0, 0), [pathname]);
  return null;
}

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <ScrollToTop />
      <NavBar />
      <main id="main" className="flex-1" tabIndex={-1}>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="search" element={<SearchPage />} />
        <Route path="video/:id" element={<VideoDetailPage />} />
        <Route path="compare" element={<ComparePage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="library" element={<LibraryPage />} />
        <Route path="benchmarks" element={<BenchmarksPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

const client = makeQueryClient();

export function App() {
  return (
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
