.bail on
.headers on
.mode markdown
.nullvalue —

.print '# Local game detail'
.print
.print '## Game'
WITH event_times AS (
    SELECT
        game_id,
        MIN(CASE json_extract(event_json, '$.event_type')
            WHEN 'game_created' THEN json_extract(event_json, '$.occurred_at') END) AS started_at,
        MAX(CASE json_extract(event_json, '$.event_type')
            WHEN 'game_completed' THEN json_extract(event_json, '$.occurred_at') END) AS completed_at
    FROM public_events
    WHERE game_id = @game_id
    GROUP BY game_id
), decoded AS (
    SELECT
        games.game_id,
        json_extract(session_json, '$.phase.kind') AS phase,
        json_extract(session_json, '$.phase.outcome') AS outcome,
        json_extract(session_json, '$.human_role') AS human_role,
        json_extract(session_json, '$.engine_state.total_scores.queen') AS queen_total,
        json_extract(session_json, '$.engine_state.total_scores.king') AS king_total,
        json_extract(session_json, '$.policy_session.policy_id') AS policy_id,
        json_extract(session_json, '$.policy_session.policy_version') AS policy_version,
        json_extract(session_json, '$.policy_session.inference_profile') AS inference_profile,
        started_at,
        completed_at
    FROM games
    LEFT JOIN event_times USING (game_id)
    WHERE games.game_id = @game_id
)
SELECT
    game_id,
    started_at AS started_utc,
    completed_at AS completed_utc,
    phase,
    human_role,
    outcome,
    CASE human_role WHEN 'queen' THEN queen_total ELSE king_total END AS human_total,
    CASE human_role WHEN 'queen' THEN king_total ELSE queen_total END AS dracula_total,
    policy_id,
    policy_version,
    inference_profile
FROM decoded;

.print
.print '## Rounds'
WITH selected AS (
    SELECT
        json_extract(session_json, '$.human_role') AS human_role,
        session_json
    FROM games
    WHERE game_id = @game_id
)
SELECT
    json_extract(round.value, '$.round_number') AS round,
    json_extract(round.value, '$.dealer') AS dealer,
    json_extract(round.value, '$.round_scores.' || selected.human_role) AS human_score,
    json_extract(
        round.value,
        '$.round_scores.' || CASE selected.human_role
            WHEN 'queen' THEN 'king'
            ELSE 'queen'
        END
    ) AS dracula_score
FROM selected
JOIN json_each(selected.session_json, '$.engine_state.completed_rounds') AS round
ORDER BY json_extract(round.value, '$.round_number');
