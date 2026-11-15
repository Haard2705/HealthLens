import Link from "next/link";
import { AtAGlance } from "@/components/AtAGlance";

const WHAT_YOU_CAN_DO = [
  {
    title: "Explore the data",
    body: "See how common each condition is, by age, weight, physical activity and sex.",
    href: "/explore",
    cta: "Open Explore",
  },
  {
    title: "See how each model was judged",
    body: "Every condition is scored on people the model never saw, and a written rule decides whether it may show a personal number.",
    href: "/models",
    cta: "Open Model results",
  },
  {
    title: "Know who sees what",
    body: "Patients see their own record, doctors only their assigned patients, and every look is logged.",
    href: "/login",
    cta: "Try the roles",
  },
];

const WHO_IT_IS_FOR = [
  {
    title: "Patient",
    body: "Sees only their own record: conditions, medications and trends, in plainer language.",
    cta: "Sign in as a demo patient",
    role: "patient",
  },
  {
    title: "Doctor",
    body: "Sees only the patients assigned to them, with history, medications and observation trends.",
    cta: "Sign in as a demo doctor",
    role: "doctor",
  },
  {
    title: "Admin",
    body: "Manages users and the system. Can open a record, but every open is written to an audit log.",
    cta: "Sign in as a demo admin",
    role: "admin",
  },
];

const HOW_IT_WORKS = [
  {
    n: "1",
    title: "Survey data",
    body: "Hundreds of thousands of adults answered the CDC BRFSS national health survey.",
  },
  {
    n: "2",
    title: "A model per condition",
    body: "Each condition is modelled two ways and scored on data the model never saw.",
  },
  {
    n: "3",
    title: "Reasons you can read",
    body: "An estimate is shown with what raised it and what lowered it, in plain words.",
  },
  {
    n: "4",
    title: "Safety checks",
    body: "Weak or sensitive models never show a personal number, only population patterns.",
  },
];

export default function HomePage() {
  return (
    <div className="home">
      <section className="h-hero">
        <span className="pill acc" style={{ alignSelf: "flex-start" }}>
          Public health, made readable
        </span>
        <h1>See what drives health risk, in plain language.</h1>
        <p className="lead">
          HealthLens turns a national health survey into clear charts and reasons that patients,
          doctors and administrators can actually use.
        </p>
        <div className="row">
          <Link className="btn primary" href="/explore">
            Explore the data
          </Link>
          <Link className="btn" href="/login">
            Sign in to the demo
          </Link>
        </div>
        <p className="small muted">
          Survey numbers come from the real BRFSS file. Every patient is invented.
        </p>
      </section>

      <section className="h-sec">
        <h2>What you can do</h2>
        <div className="h-cards">
          {WHAT_YOU_CAN_DO.map((card) => (
            <div className="h-card" key={card.title}>
              <h3>{card.title}</h3>
              <p>{card.body}</p>
              <Link className="btn" href={card.href}>
                {card.cta}
              </Link>
            </div>
          ))}
        </div>
      </section>

      <section className="h-sec" id="roles">
        <h2>Who it is for</h2>
        <div className="h-cards">
          {WHO_IT_IS_FOR.map((card) => (
            <div className="h-card" key={card.title}>
              <h3>{card.title}</h3>
              <p>{card.body}</p>
              <Link className="btn" href={`/login?as=${card.role}`}>
                {card.cta}
              </Link>
            </div>
          ))}
        </div>
      </section>

      <section className="h-sec">
        <h2>How it works</h2>
        <div className="steps">
          {HOW_IT_WORKS.map((step) => (
            <div className="step" key={step.n}>
              <span className="n">{step.n}</span>
              <strong>{step.title}</strong>
              <span className="d">{step.body}</span>
            </div>
          ))}
        </div>
      </section>

      <AtAGlance />

      <section className="alert">
        <strong>Please note.</strong> HealthLens shows patterns in survey data. It is not a
        diagnosis and gives no medical advice. Groups with fewer than 30 respondents are hidden
        rather than shown as a misleading number, and sensitive conditions are kept to population
        analytics whatever their model score.
      </section>
    </div>
  );
}
