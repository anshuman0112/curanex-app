-- Curanex per-user data in Supabase. Run once: Supabase dashboard -> SQL Editor -> New query -> paste all -> Run.
-- Row level security is ON with no policies, so only the Curanex server (secret key) can read or write; the public key cannot.
-- user_id / user_email = the Curanex account (local login). Uploaded files live in the private Storage bucket "curanex-uploads".

create table if not exists chat_messages (
  id bigint generated always as identity primary key,
  user_id bigint not null, user_email text, session_id text,
  role text not null,               -- 'user' or 'assistant'
  content text,                     -- what was said (assistant: plain text of the reply)
  blocks jsonb,                     -- the full reply as shown in the app (cards, model results, forms)
  created_at timestamptz not null default now()
);
create table if not exists uploads (
  id bigint generated always as identity primary key,
  user_id bigint not null, user_email text,
  image_id text, file_path text, kind text, body_part text, model_id text, findings jsonb,
  created_at timestamptz not null default now()
);
create table if not exists reports (
  id bigint generated always as identity primary key,
  user_id bigint not null, user_email text,
  file_path text, text text, lab_values jsonb,
  created_at timestamptz not null default now()
);
create table if not exists model_results (
  id bigint generated always as identity primary key,
  user_id bigint not null, user_email text, session_id text,
  model_id text, model_name text, label text, flag text, probabilities jsonb, inputs jsonb, verified boolean,
  created_at timestamptz not null default now()
);
create table if not exists cases (
  id bigint generated always as identity primary key,
  user_id bigint not null, user_email text, local_case_id bigint,
  title text, data jsonb,
  created_at timestamptz not null default now()
);
create table if not exists doctor_plans (
  id bigint generated always as identity primary key,
  patient_id bigint, doctor_id bigint, local_case_id bigint, plan jsonb,
  created_at timestamptz not null default now()
);

create index if not exists chat_messages_user on chat_messages (user_id, created_at desc);
create index if not exists uploads_user on uploads (user_id, created_at desc);
create index if not exists reports_user on reports (user_id, created_at desc);
create index if not exists model_results_user on model_results (user_id, created_at desc);
create index if not exists cases_user on cases (user_id, created_at desc);

alter table chat_messages enable row level security;
alter table uploads enable row level security;
alter table reports enable row level security;
alter table model_results enable row level security;
alter table cases enable row level security;
alter table doctor_plans enable row level security;
