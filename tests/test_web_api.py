"""The web UI and its JSON endpoints (leave application form, my leave, team view)."""
from app.db import repository as repo


def get(client, path, employee_id="E3004"):
    return client.get(path, headers={"X-Employee-Id": employee_id})


def post(client, path, body, employee_id="E3004"):
    return client.post(path, json=body, headers={"X-Employee-Id": employee_id})


def test_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Veltrix PeopleOps" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_identity_required(client):
    assert client.get("/api/me").status_code == 401
    assert get(client, "/api/me", "E9999").status_code == 401


def test_me_has_no_phone_and_masked_email(client, settings):
    with repo.session(settings.db_path) as conn:
        me = repo.get_employee(conn, "E3004")
    body = get(client, "/api/me").json()
    assert body["employee_id"] == "E3004" and body["role"] == "employee" and body["today"] == "2026-10-15"
    assert me.phone not in str(body) and me.email not in str(body)
    assert body["email"].startswith(me.email[0] + "***@")


def test_documents_only_lists_readable_titles(client):
    titles = get(client, "/api/documents").json()
    assert "POL-LEAVE-GLOBAL-V2" in titles
    assert "HR-COMP-BANDS-2026" not in titles and "HB-MANAGER" not in titles
    assert "ADD-LEAVE-DE" not in titles  # other region


def test_holidays_for_own_region(client):
    days = {h["date"] for h in get(client, "/api/holidays").json()}
    assert "2026-11-26" in days  # US Thanksgiving
    assert "2026-11-09" not in days  # India-only holiday


def test_preview_never_writes(client, settings):
    body = post(client, "/api/leave/preview",
                {"leave_type": "annual", "start_date": "2026-11-23", "end_date": "2026-11-27"}, "E3002").json()
    assert body["ok"] is True and body["working_days"] == 3
    assert [h["date"] for h in body["holidays"]] == ["2026-11-26", "2026-11-27"]
    with repo.session(settings.db_path) as conn:
        assert not repo.active_overlaps(conn, "E3002", "2026-11-23", "2026-11-27")


def test_preview_returns_rule_reason(client):
    body = post(client, "/api/leave/preview",
                {"leave_type": "annual", "start_date": "2026-11-16", "end_date": "2026-11-20"}, "E2004").json()
    assert body["ok"] is False and "balance" in body["reason"]


def test_apply_then_cancel(client, settings):
    applied = post(client, "/api/leave/apply",
                   {"leave_type": "annual", "start_date": "2026-11-02", "end_date": "2026-11-06"})
    assert applied.status_code == 200
    result = applied.json()
    assert result["status"] == "pending" and result["working_days"] == 5 and result["balance_after"] == 3.5

    requests = get(client, "/api/leave/requests").json()
    mine = next(r for r in requests if r["request_id"] == result["request_id"])
    assert mine["cancellable"] is True
    assert next(r for r in requests if r["start_date"] == "2026-07-06")["cancellable"] is False  # in the past

    cancelled = post(client, "/api/leave/cancel", {"request_id": result["request_id"]})
    assert cancelled.status_code == 200 and cancelled.json()["balance_after"] == 8.5


def test_apply_rejection_is_422_with_reason(client):
    res = post(client, "/api/leave/apply",
               {"leave_type": "annual", "start_date": "2026-11-07", "end_date": "2026-11-08"})
    assert res.status_code == 422 and "no working days" in res.json()["detail"]


def test_cannot_cancel_someone_elses_request(client):
    res = post(client, "/api/leave/cancel", {"request_id": "e842997d-a9f5-407c-8ece-78b06f29a9f1"})
    assert res.status_code == 422 and "couldn't find" in res.json()["detail"]


def test_team_view_is_managers_only(client):
    assert get(client, "/api/team/leave?start_date=2026-11-01&end_date=2026-11-30").status_code == 403
    rows = get(client, "/api/team/leave?start_date=2026-11-01&end_date=2026-11-30", "E3002").json()
    assert rows and {r["employee_id"] for r in rows} <= {"E3004", "E3005", "E3006"}
