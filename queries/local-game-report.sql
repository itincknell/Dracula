.bail on
.headers on
.mode markdown
.nullvalue —

DROP VIEW IF EXISTS temp.local_game_results;
CREATE TEMP VIEW local_game_results AS
WITH event_times AS (
    SELECT
        game_id,
        MIN(
            CASE json_extract(event_json, '$.event_type')
                WHEN 'game_created' THEN json_extract(event_json, '$.occurred_at')
            END
        ) AS started_at,
        MAX(
            CASE json_extract(event_json, '$.event_type')
                WHEN 'game_completed' THEN json_extract(event_json, '$.occurred_at')
            END
        ) AS completed_at
    FROM public_events
    GROUP BY game_id
), decoded AS (
    SELECT
        games.game_id,
        json_extract(games.session_json, '$.phase.kind') AS phase,
        json_extract(games.session_json, '$.phase.outcome') AS outcome,
        json_extract(games.session_json, '$.human_role') AS human_role,
        json_extract(games.session_json, '$.engine_state.total_scores.queen') AS queen_total,
        json_extract(games.session_json, '$.engine_state.total_scores.king') AS king_total,
        json_extract(games.session_json, '$.policy_session.policy_id') AS policy_id,
        json_extract(games.session_json, '$.policy_session.policy_version') AS policy_version,
        json_extract(games.session_json, '$.policy_session.inference_profile') AS inference_profile,
        event_times.started_at,
        event_times.completed_at
    FROM games
    LEFT JOIN event_times USING (game_id)
)
SELECT
    game_id,
    phase,
    outcome,
    human_role,
    CASE human_role WHEN 'queen' THEN queen_total ELSE king_total END AS human_total,
    CASE human_role WHEN 'queen' THEN king_total ELSE queen_total END AS dracula_total,
    policy_id,
    policy_version,
    inference_profile,
    started_at,
    completed_at,
    CASE
        WHEN completed_at IS NULL THEN NULL
        ELSE ROUND((julianday(completed_at) - julianday(started_at)) * 86400)
    END AS duration_seconds
FROM decoded;

DROP VIEW IF EXISTS temp.local_round_results;
CREATE TEMP VIEW local_round_results AS
SELECT
    result.game_id,
    result.human_role,
    json_extract(round.value, '$.round_number') AS round_number,
    json_extract(round.value, '$.dealer') AS dealer,
    json_extract(
        round.value,
        '$.round_scores.' || result.human_role
    ) AS human_round_score,
    json_extract(
        round.value,
        '$.round_scores.' || CASE result.human_role
            WHEN 'queen' THEN 'king'
            ELSE 'queen'
        END
    ) AS dracula_round_score
FROM local_game_results AS result
JOIN games USING (game_id)
JOIN json_each(games.session_json, '$.engine_state.completed_rounds') AS round
WHERE result.phase = 'game_complete';

.print '# Local gameplay report'
.print
.print '## Collection status'
SELECT
    COUNT(*) AS started_games,
    SUM(phase = 'game_complete') AS completed_games,
    SUM(phase <> 'game_complete') AS in_progress_games,
    MIN(started_at) AS first_started_utc,
    MAX(completed_at) AS last_completed_utc
FROM local_game_results;

.print
.print '## Completed-game results'
SELECT
    COUNT(*) AS games,
    SUM(outcome = 'human') AS human_wins,
    SUM(outcome = 'opponent') AS dracula_wins,
    SUM(outcome = 'tie') AS ties,
    CASE COUNT(*)
        WHEN 0 THEN '—'
        ELSE printf('%.1f%%', 100.0 * SUM(outcome = 'human') / COUNT(*))
    END AS human_win_rate,
    ROUND(AVG(human_total), 1) AS avg_human_score,
    ROUND(AVG(dracula_total), 1) AS avg_dracula_score,
    ROUND(AVG(human_total - dracula_total), 1) AS avg_human_margin
FROM local_game_results
WHERE phase = 'game_complete';

.print
.print '## Results by human role'
SELECT
    human_role,
    COUNT(*) AS games,
    SUM(outcome = 'human') AS human_wins,
    SUM(outcome = 'opponent') AS dracula_wins,
    SUM(outcome = 'tie') AS ties,
    printf('%.1f%%', 100.0 * SUM(outcome = 'human') / COUNT(*)) AS human_win_rate,
    ROUND(AVG(human_total - dracula_total), 1) AS avg_human_margin
FROM local_game_results
WHERE phase = 'game_complete'
GROUP BY human_role
ORDER BY human_role;

.print
.print '## Results by opponent'
SELECT
    policy_id,
    policy_version,
    inference_profile,
    COUNT(*) AS games,
    SUM(outcome = 'human') AS human_wins,
    SUM(outcome = 'opponent') AS dracula_wins,
    SUM(outcome = 'tie') AS ties,
    printf('%.1f%%', 100.0 * SUM(outcome = 'human') / COUNT(*)) AS human_win_rate,
    ROUND(AVG(human_total - dracula_total), 1) AS avg_human_margin
FROM local_game_results
WHERE phase = 'game_complete'
GROUP BY policy_id, policy_version, inference_profile
ORDER BY policy_id, policy_version, inference_profile;

.print
.print '## Round results from completed games'
SELECT
    COUNT(*) AS rounds,
    SUM(human_round_score > dracula_round_score) AS human_round_wins,
    SUM(human_round_score < dracula_round_score) AS dracula_round_wins,
    SUM(human_round_score = dracula_round_score) AS tied_rounds,
    ROUND(AVG(human_round_score), 1) AS avg_human_round_score,
    ROUND(AVG(dracula_round_score), 1) AS avg_dracula_round_score,
    ROUND(AVG(human_round_score - dracula_round_score), 1) AS avg_human_round_margin
FROM local_round_results;

.print
.print '## Completed games'
SELECT
    game_id,
    completed_at AS completed_utc,
    human_role,
    outcome,
    human_total,
    dracula_total,
    human_total - dracula_total AS human_margin,
    policy_version,
    duration_seconds
FROM local_game_results
WHERE phase = 'game_complete'
ORDER BY completed_at DESC, game_id;

.print
.print '## In-progress games'
SELECT
    game_id,
    started_at AS started_utc,
    human_role,
    phase,
    policy_version
FROM local_game_results
WHERE phase <> 'game_complete'
ORDER BY started_at DESC, game_id;
