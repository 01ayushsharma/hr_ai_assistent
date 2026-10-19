from datetime import date

import pytest

from app.db import repository as repo
from app.tools.working_days import applicable_calendars, count_working_days, holidays_in_range


def d(s: str) -> date:
    return date.fromisoformat(s)


class TestPureCount:
    def test_full_week(self):
        assert count_working_days(d("2026-11-02"), d("2026-11-06"), set()) == 5

    def test_weekend_only(self):
        assert count_working_days(d("2026-11-07"), d("2026-11-08"), set()) == 0

    def test_range_spanning_weekend(self):
        # Thu, Fri, (Sat, Sun), Mon, Tue
        assert count_working_days(d("2026-11-05"), d("2026-11-10"), set()) == 4

    def test_holiday_on_weekday_is_excluded(self):
        assert count_working_days(d("2026-11-02"), d("2026-11-06"), {d("2026-11-04")}) == 4

    def test_holiday_on_weekend_not_subtracted_twice(self):
        assert count_working_days(d("2026-11-02"), d("2026-11-08"), {d("2026-11-07")}) == 5

    def test_single_day(self):
        assert count_working_days(d("2026-11-04"), d("2026-11-04"), set()) == 1


class TestCalendarsFromCorpus:
    def test_calendar_choice_follows_section5(self, kb):
        # Both India calendars are eligible in 2026; the 2026 one wins on recency.
        assert applicable_calendars(kb, "employee", "IN", d("2026-03-01")) == ["CAL-HOL-IN-2026"]
        assert applicable_calendars(kb, "employee", "IN", d("2025-06-01")) == ["CAL-HOL-IN-2025"]
        assert applicable_calendars(kb, "employee", "DE", d("2026-12-01")) == ["CAL-HOL-DE-2026"]

    def test_restricted_holidays_are_not_closures(self, kb):
        holidays = holidays_in_range(kb, "employee", "IN", d("2026-03-01"), d("2026-03-31"))
        assert d("2026-03-19") in holidays  # public holiday
        assert d("2026-03-04") not in holidays  # restricted holiday: office open

    @pytest.mark.parametrize("region,start,end,expected", [
        ("IN", "2026-11-09", "2026-11-13", 4),   # Monday public holiday
        ("IN", "2026-10-19", "2026-10-23", 4),   # Tuesday public holiday
        ("DE", "2026-12-21", "2026-12-31", 6),   # three weekday holidays + weekend
        ("US", "2026-12-21", "2026-12-31", 7),   # two weekday holidays + weekend
        ("US", "2026-11-23", "2026-11-27", 3),   # Thanksgiving + day after
        ("DE", "2026-10-03", "2026-10-04", 0),   # holiday on a Saturday + Sunday
    ])
    def test_region_ranges(self, kb, region, start, end, expected):
        holidays = holidays_in_range(kb, "employee", region, d(start), d(end))
        assert count_working_days(d(start), d(end), set(holidays)) == expected

    def test_matches_every_seeded_request(self, kb, settings):
        """The working_days stored in the database agree with our calculation."""
        with repo.session(settings.db_path) as conn:
            rows = conn.execute(
                "SELECT r.start_date, r.end_date, r.working_days, e.region, e.role "
                "FROM leave_requests r JOIN employees e USING (employee_id)").fetchall()
        assert rows
        for row in rows:
            start, end = d(row["start_date"]), d(row["end_date"])
            holidays = holidays_in_range(kb, row["role"], row["region"], start, end)
            assert count_working_days(start, end, set(holidays)) == row["working_days"], dict(row)
