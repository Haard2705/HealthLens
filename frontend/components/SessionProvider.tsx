"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { ApiError, apiGet, apiSend } from "@/lib/api";
import type { User } from "@/lib/types";

export interface Session {
  user: User | null;
  /** True until the first `/auth/me` answer comes back. */
  loading: boolean;
  refresh: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<User>;
  signUp: (fullName: string, email: string, password: string) => Promise<User>;
  signOut: () => Promise<void>;
}

const SessionContext = createContext<Session | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setUser(await apiGet<User>("/auth/me"));
    } catch (error: unknown) {
      // 401 simply means nobody is signed in; anything else also leaves us signed out.
      if (!(error instanceof ApiError)) setUser(null);
      else setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const signIn = useCallback(async (email: string, password: string) => {
    const signedIn = await apiSend<User>("POST", "/auth/login", { email, password });
    setUser(signedIn);
    setLoading(false);
    return signedIn;
  }, []);

  const signUp = useCallback(async (fullName: string, email: string, password: string) => {
    const created = await apiSend<User>("POST", "/auth/register", {
      full_name: fullName,
      email,
      password,
    });
    setUser(created);
    setLoading(false);
    return created;
  }, []);

  const signOut = useCallback(async () => {
    try {
      await apiSend<void>("POST", "/auth/logout");
    } finally {
      setUser(null);
    }
  }, []);

  return (
    <SessionContext.Provider value={{ user, loading, refresh, signIn, signUp, signOut }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("useSession must be used inside <SessionProvider>");
  return session;
}
