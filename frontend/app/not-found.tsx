import Link from "next/link";

export default function NotFound() {
  return (
    <div className="stack" style={{ maxWidth: 560 }}>
      <div className="card">
        <span className="pill">404</span>
        <h1 className="pt" style={{ marginTop: 10 }}>
          There is no page here
        </h1>
        <p className="muted" style={{ marginTop: 6 }}>
          The link may be out of date. Everything public starts from the home page.
        </p>
        <Link className="btn primary" style={{ marginTop: 14 }} href="/">
          Go to the home page
        </Link>
      </div>
    </div>
  );
}
