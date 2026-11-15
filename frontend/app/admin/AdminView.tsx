"use client";

import { useCallback, useState } from "react";
import { useSession } from "@/components/SessionProvider";
import { ErrorNote, Loading, SignInRequired, WrongRole } from "@/components/States";
import { apiGet, apiSend, errorMessage } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import { count, humanise } from "@/lib/format";
import type { AuditEvent, Items, Role, SystemInfo, User } from "@/lib/types";

type AuditFilter = "all" | "reads" | "denied";

function SystemCards({ info }: { info: SystemInfo }) {
  const cards = [
    { label: "Patient data", value: humanise(info.data_source), note: `${info.patient_count} demo patients` },
    { label: "Accounts", value: count(info.user_count), note: "invite-only; nobody self-registers" },
    {
      label: "Prediction enabled",
      value: `${info.conditions_prediction_enabled} of ${info.conditions_total}`,
      note: "the rest show population analytics only",
    },
    { label: "Environment", value: info.environment, note: `demo domain ${info.demo_domain}` },
  ];
  return (
    <div className="grid">
      {cards.map((card) => (
        <div className="card kpi" key={card.label}>
          <span className="small muted">{card.label}</span>
          <span className="v" style={{ fontSize: 20 }}>
            {card.value}
          </span>
          <span className="small muted">{card.note}</span>
        </div>
      ))}
    </div>
  );
}

export function AdminView() {
  const { user, loading: sessionLoading } = useSession();
  const isAdmin = user?.role === "admin";
  const [filter, setFilter] = useState<AuditFilter>("all");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteName, setInviteName] = useState("");
  const [inviteRole, setInviteRole] = useState<Role>("doctor");
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [inviteOk, setInviteOk] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadUsers = useCallback(
    (signal: AbortSignal) =>
      isAdmin
        ? apiGet<Items<User>>("/admin/users", undefined, signal)
        : Promise.resolve({ items: [] }),
    [isAdmin],
  );
  const users = useResource(loadUsers);

  const loadAudit = useCallback(
    (signal: AbortSignal) =>
      isAdmin
        ? apiGet<Items<AuditEvent>>(
            "/admin/audit",
            {
              outcome: filter === "denied" ? "denied" : undefined,
              action: filter === "reads" ? "patient.read" : undefined,
              limit: 100,
            },
            signal,
          )
        : Promise.resolve({ items: [] }),
    [isAdmin, filter],
  );
  const audit = useResource(loadAudit);

  const loadSystem = useCallback(
    (signal: AbortSignal) =>
      isAdmin
        ? apiGet<SystemInfo>("/admin/system", undefined, signal)
        : Promise.reject(new Error("not an admin")),
    [isAdmin],
  );
  const system = useResource(loadSystem);

  if (sessionLoading) return <Loading what="your account" />;
  if (!user) return <SignInRequired what="the admin area" />;
  if (!isAdmin) return <WrongRole expected="admin" />;

  async function invite(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setInviteError(null);
    setInviteOk(null);
    try {
      const created = await apiSend<User>("POST", "/admin/users/invite", {
        email: inviteEmail.trim(),
        role: inviteRole,
        display_name: inviteName.trim(),
      });
      setInviteOk(`Invited ${created.email}. The account stays inactive until it is accepted.`);
      setInviteEmail("");
      setInviteName("");
      users.reload();
      audit.reload();
    } catch (caught: unknown) {
      setInviteError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }

  async function toggleActive(target: User) {
    setBusy(true);
    setInviteError(null);
    try {
      await apiSend<User>("PATCH", `/admin/users/${encodeURIComponent(target.id)}`, {
        active: !target.active,
      });
      users.reload();
      audit.reload();
    } catch (caught: unknown) {
      setInviteError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div>
        <h1 className="pt">Users and audit</h1>
        <p className="muted">
          Accounts are invite-only: doctors and admins cannot register themselves. Every read or
          change of patient data, and every refused attempt, lands in the audit log below.
        </p>
      </div>

      {system.data && <SystemCards info={system.data} />}
      {system.error && <ErrorNote message={system.error} onRetry={system.reload} />}

      <form className="card stack" onSubmit={invite}>
        <h2 style={{ fontSize: 16 }}>Invite a user</h2>
        <div className="row" style={{ alignItems: "flex-end" }}>
          <label className="f">
            Email
            <input
              type="email"
              required
              value={inviteEmail}
              onChange={(event) => setInviteEmail(event.target.value)}
              placeholder="name@clinic.org"
            />
          </label>
          <label className="f">
            Display name
            <input
              type="text"
              required
              value={inviteName}
              onChange={(event) => setInviteName(event.target.value)}
              placeholder="Dr. Rivera"
            />
          </label>
          <label className="f">
            Role
            <select
              value={inviteRole}
              onChange={(event) => setInviteRole(event.target.value as Role)}
            >
              <option value="doctor">Doctor</option>
              <option value="admin">Admin</option>
            </select>
          </label>
          <button className="btn primary" type="submit" disabled={busy}>
            {busy ? "Working…" : "Send invite"}
          </button>
        </div>
        {inviteError && (
          <p className="small" role="alert" style={{ color: "var(--bad)" }}>
            {inviteError}
          </p>
        )}
        {inviteOk && (
          <p className="small" role="status" style={{ color: "var(--ok)" }}>
            {inviteOk}
          </p>
        )}
        <p className="small muted">
          Patient accounts need an existing patient id, so they are created with the seed data
          rather than here.
        </p>
      </form>

      <div className="stack" style={{ gap: 8 }}>
        <h2 style={{ fontSize: 16 }}>Accounts</h2>
        {users.loading && <Loading what="the user list" />}
        {users.error && <ErrorNote message={users.error} onRetry={users.reload} />}
        {users.data && (
          <div className="card tw">
            <table>
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>
                    <span className="sr">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {users.data.items.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <strong>{item.display_name}</strong>
                      <div className="small muted mono">{item.email}</div>
                    </td>
                    <td>
                      <span className="pill acc">{item.role}</span>
                    </td>
                    <td>
                      <span className={`pill ${item.active ? "ok" : "warn"}`}>
                        {item.active ? "active" : "inactive"}
                      </span>
                    </td>
                    <td className="r">
                      <button
                        className="btn small"
                        type="button"
                        disabled={busy || item.id === user.id}
                        onClick={() => toggleActive(item)}
                      >
                        {item.active ? "Deactivate" : "Activate"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="stack" style={{ gap: 8 }}>
        <div className="row between">
          <h2 style={{ fontSize: 16 }}>Audit log</h2>
          <label className="f">
            Show
            <select
              value={filter}
              onChange={(event) => setFilter(event.target.value as AuditFilter)}
            >
              <option value="all">All events</option>
              <option value="reads">Record reads</option>
              <option value="denied">Denied attempts</option>
            </select>
          </label>
        </div>
        <p className="small muted">
          Append-only. It records who, what and when — never clinical content. Sign in as the
          doctor, try the unassigned patient, then come back here: the denied event is waiting.
        </p>
        {audit.loading && <Loading what="the audit log" />}
        {audit.error && <ErrorNote message={audit.error} onRetry={audit.reload} />}
        {audit.data && (
          <div className="card tw">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Actor</th>
                  <th>Action</th>
                  <th>Resource</th>
                  <th>Patient</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {audit.data.items.length === 0 && (
                  <tr>
                    <td colSpan={6} className="muted">
                      No events match that filter.
                    </td>
                  </tr>
                )}
                {audit.data.items.map((event) => (
                  <tr key={event.id} className={event.outcome === "denied" ? "deny" : undefined}>
                    <td className="mono small">{event.timestamp.slice(11, 19)}</td>
                    <td>
                      {event.actor_id ?? "anonymous"}{" "}
                      <span className="small muted">{event.actor_role ?? ""}</span>
                    </td>
                    <td className="mono small">{event.action}</td>
                    <td className="small">{event.resource_type}</td>
                    <td className="mono small">{event.patient_id ?? "—"}</td>
                    <td>
                      <span className={`pill ${event.outcome === "allowed" ? "ok" : "bad"}`}>
                        {event.outcome}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
