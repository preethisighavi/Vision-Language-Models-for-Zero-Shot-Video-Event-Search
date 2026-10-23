import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="wrap-marketing py-24 text-center">
      <h1 className="h-page">Nothing here.</h1>
      <p className="lead mt-4">That page doesn't exist.</p>
      <Link to="/" className="btn-primary mt-8">
        Go home
      </Link>
    </div>
  );
}
