import unittest
import pandas as pd
import numpy as np
from paginator import DataFramePaginator, PageResult


class TestPageResult(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

    def test_to_dict_page_mode(self):
        result = PageResult(
            data=self.df,
            total=100,
            page=1,
            page_size=10,
            total_pages=10,
            has_next=True,
            has_prev=False,
        )
        d = result.to_dict()
        self.assertEqual(d["total"], 100)
        self.assertEqual(d["page"], 1)
        self.assertEqual(d["page_size"], 10)
        self.assertEqual(d["total_pages"], 10)
        self.assertTrue(d["has_next"])
        self.assertFalse(d["has_prev"])
        self.assertEqual(len(d["data"]), 3)

    def test_to_dict_cursor_mode(self):
        result = PageResult(
            data=self.df,
            total=100,
            cursor=5,
            limit=10,
            has_next=True,
            has_prev=True,
            next_cursor=15,
            prev_cursor=3,
        )
        d = result.to_dict()
        self.assertEqual(d["total"], 100)
        self.assertEqual(d["cursor"], 5)
        self.assertEqual(d["limit"], 10)
        self.assertEqual(d["next_cursor"], 15)
        self.assertEqual(d["prev_cursor"], 3)

    def test_repr(self):
        result = PageResult(data=self.df, total=100, page=1, page_size=10, total_pages=10)
        self.assertIn("page", repr(result))
        self.assertIn("100", repr(result))


class TestDataFramePaginatorInit(unittest.TestCase):
    def test_init_with_valid_df(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        paginator = DataFramePaginator(df)
        self.assertEqual(len(paginator.df), 3)

    def test_init_with_none(self):
        with self.assertRaises(ValueError):
            DataFramePaginator(None)

    def test_init_copies_df(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        paginator = DataFramePaginator(df)
        df.loc[0, "id"] = 999
        self.assertEqual(paginator.df.loc[0, "id"], 1)


class TestPaginateByPage(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": range(1, 101), "value": range(100, 200)})
        self.paginator = DataFramePaginator(self.df)

    def test_first_page(self):
        result = self.paginator.paginate_by_page(page=1, page_size=10, sort_by="id")
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.page, 1)
        self.assertEqual(result.page_size, 10)
        self.assertEqual(result.total, 100)
        self.assertEqual(result.total_pages, 10)
        self.assertFalse(result.has_prev)
        self.assertTrue(result.has_next)
        self.assertEqual(result.data["id"].iloc[0], 1)
        self.assertEqual(result.data["id"].iloc[-1], 10)

    def test_middle_page(self):
        result = self.paginator.paginate_by_page(page=5, page_size=10, sort_by="id")
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.page, 5)
        self.assertEqual(result.total_pages, 10)
        self.assertTrue(result.has_prev)
        self.assertTrue(result.has_next)
        self.assertEqual(result.data["id"].iloc[0], 41)
        self.assertEqual(result.data["id"].iloc[-1], 50)

    def test_last_page(self):
        result = self.paginator.paginate_by_page(page=10, page_size=10, sort_by="id")
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.page, 10)
        self.assertTrue(result.has_prev)
        self.assertFalse(result.has_next)
        self.assertEqual(result.data["id"].iloc[0], 91)
        self.assertEqual(result.data["id"].iloc[-1], 100)

    def test_page_beyond_total(self):
        result = self.paginator.paginate_by_page(page=20, page_size=10, sort_by="id")
        self.assertEqual(len(result.data), 0)
        self.assertEqual(result.page, 20)
        self.assertTrue(result.has_prev)
        self.assertFalse(result.has_next)

    def test_page_size_not_divisible(self):
        df = pd.DataFrame({"id": range(1, 26)})
        paginator = DataFramePaginator(df)
        result = paginator.paginate_by_page(page=1, page_size=10, sort_by="id")
        self.assertEqual(result.total_pages, 3)
        result = paginator.paginate_by_page(page=3, page_size=10, sort_by="id")
        self.assertEqual(len(result.data), 5)

    def test_sort_descending(self):
        result = self.paginator.paginate_by_page(
            page=1, page_size=5, sort_by="id", ascending=False
        )
        self.assertEqual(result.data["id"].iloc[0], 100)
        self.assertEqual(result.data["id"].iloc[-1], 96)

    def test_sort_by_another_column(self):
        np.random.seed(0)
        df = pd.DataFrame({"id": range(1, 21), "score": np.random.randint(0, 100, 20)})
        paginator = DataFramePaginator(df)
        result = paginator.paginate_by_page(page=1, page_size=5, sort_by="score", ascending=False)
        scores = result.data["score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_invalid_page(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_page(page=0, page_size=10)
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_page(page=-1, page_size=10)

    def test_invalid_page_size(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_page(page=1, page_size=0)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"id": []})
        paginator = DataFramePaginator(df)
        result = paginator.paginate_by_page(page=1, page_size=10, sort_by="id")
        self.assertEqual(result.total, 0)
        self.assertEqual(result.total_pages, 0)
        self.assertEqual(len(result.data), 0)
        self.assertFalse(result.has_next)
        self.assertFalse(result.has_prev)

    def test_no_sort_by(self):
        result = self.paginator.paginate_by_page(page=1, page_size=5)
        self.assertEqual(len(result.data), 5)


class TestPaginateByCursor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": range(1, 101), "value": range(100, 200)})
        self.paginator = DataFramePaginator(self.df)

    def test_first_page_no_cursor(self):
        result = self.paginator.paginate_by_cursor(cursor=None, limit=10, sort_by="id", count_total=True)
        self.assertEqual(len(result.data), 10)
        self.assertIsNone(result.cursor)
        self.assertEqual(result.total, 100)
        self.assertFalse(result.has_prev)
        self.assertTrue(result.has_next)
        self.assertIsNotNone(result.next_cursor)
        self.assertIsNone(result.prev_cursor)
        self.assertEqual(result.data["id"].iloc[0], 1)
        self.assertEqual(result.data["id"].iloc[-1], 10)
        self.assertEqual(result.next_cursor, 10)

    def test_next_page_with_cursor(self):
        result = self.paginator.paginate_by_cursor(
            cursor=10, limit=10, sort_by="id", direction="next"
        )
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.cursor, 10)
        self.assertTrue(result.has_prev)
        self.assertTrue(result.has_next)
        self.assertEqual(result.data["id"].iloc[0], 11)
        self.assertEqual(result.data["id"].iloc[-1], 20)
        self.assertEqual(result.next_cursor, 20)
        self.assertEqual(result.prev_cursor, 11)

    def test_prev_page_with_cursor(self):
        result = self.paginator.paginate_by_cursor(
            cursor=21, limit=10, sort_by="id", direction="prev"
        )
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.cursor, 21)
        self.assertEqual(result.data["id"].iloc[0], 11)
        self.assertEqual(result.data["id"].iloc[-1], 20)
        self.assertEqual(result.next_cursor, 20)
        self.assertEqual(result.prev_cursor, 11)
        self.assertTrue(result.has_prev)
        self.assertTrue(result.has_next)

    def test_last_page(self):
        result = self.paginator.paginate_by_cursor(
            cursor=90, limit=10, sort_by="id", direction="next"
        )
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.data["id"].iloc[0], 91)
        self.assertEqual(result.data["id"].iloc[-1], 100)
        self.assertFalse(result.has_next)
        self.assertTrue(result.has_prev)
        self.assertIsNone(result.next_cursor)

    def test_sort_descending(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=5, sort_by="id", ascending=False
        )
        self.assertEqual(result.data["id"].iloc[0], 100)
        self.assertEqual(result.data["id"].iloc[-1], 96)
        self.assertEqual(result.next_cursor, 96)

    def test_descending_next_page(self):
        result = self.paginator.paginate_by_cursor(
            cursor=96, limit=5, sort_by="id", ascending=False, direction="next"
        )
        self.assertEqual(len(result.data), 5)
        self.assertEqual(result.data["id"].iloc[0], 95)
        self.assertEqual(result.data["id"].iloc[-1], 91)

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_cursor(cursor=None, limit=0, sort_by="id")

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_cursor(cursor=10, limit=10, sort_by="id", direction="invalid")

    def test_invalid_sort_by(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_cursor(cursor=None, limit=10, sort_by="nonexistent")

    def test_empty_dataframe(self):
        df = pd.DataFrame({"id": [], "value": []})
        paginator = DataFramePaginator(df)
        result = paginator.paginate_by_cursor(cursor=None, limit=10, sort_by="id", count_total=True)
        self.assertEqual(result.total, 0)
        self.assertEqual(len(result.data), 0)
        self.assertFalse(result.has_next)
        self.assertFalse(result.has_prev)

    def test_cursor_at_boundary(self):
        result = self.paginator.paginate_by_cursor(
            cursor=0, limit=10, sort_by="id", direction="next"
        )
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.data["id"].iloc[0], 1)


class TestUnifiedPaginate(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": range(1, 51)})
        self.paginator = DataFramePaginator(self.df)

    def test_page_mode(self):
        result = self.paginator.paginate(mode="page", page=2, page_size=5, sort_by="id")
        self.assertEqual(result.page, 2)
        self.assertEqual(len(result.data), 5)

    def test_cursor_mode(self):
        result = self.paginator.paginate(mode="cursor", cursor=None, limit=5, sort_by="id")
        self.assertIsNone(result.cursor)
        self.assertEqual(len(result.data), 5)

    def test_cursor_missing_sort_by(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate(mode="cursor", cursor=None, limit=5)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate(mode="invalid")


class TestCursorPaginationNavigation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": range(1, 31)})
        self.paginator = DataFramePaginator(self.df)

    def test_traverse_forward_all_pages(self):
        cursor = None
        all_ids = []
        page_count = 0
        while True:
            result = self.paginator.paginate_by_cursor(
                cursor=cursor, limit=10, sort_by="id", direction="next"
            )
            page_count += 1
            all_ids.extend(result.data["id"].tolist())
            if not result.has_next:
                break
            cursor = result.next_cursor

        self.assertEqual(page_count, 3)
        self.assertEqual(all_ids, list(range(1, 31)))

    def test_traverse_backward_all_pages(self):
        result = self.paginator.paginate_by_cursor(cursor=None, limit=10, sort_by="id")
        while result.has_next:
            result = self.paginator.paginate_by_cursor(
                cursor=result.next_cursor, limit=10, sort_by="id", direction="next"
            )

        all_ids = []
        page_count = 0
        cursor = result.prev_cursor
        all_ids.extend(result.data["id"].tolist())
        page_count += 1

        while cursor is not None:
            result = self.paginator.paginate_by_cursor(
                cursor=cursor, limit=10, sort_by="id", direction="prev"
            )
            page_count += 1
            all_ids = result.data["id"].tolist() + all_ids
            cursor = result.prev_cursor

        self.assertEqual(page_count, 3)
        self.assertEqual(all_ids, list(range(1, 31)))


class TestSafePagePagination(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": range(1, 31)})
        self.paginator = DataFramePaginator(self.df)

    def test_safe_mode_requires_sort_by(self):
        with self.assertRaises(ValueError):
            self.paginator.paginate_by_page(
                page=1, page_size=10, safe_mode=True
            )

    def test_safe_first_page(self):
        result = self.paginator.paginate_by_page(
            page=1, page_size=10, sort_by="id", safe_mode=True
        )
        self.assertTrue(result.safe_mode)
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.data["id"].iloc[0], 1)
        self.assertEqual(result.data["id"].iloc[-1], 10)
        self.assertEqual(result.first_seen_value, 1)
        self.assertEqual(result.last_seen_value, 10)
        self.assertTrue(result.has_next)
        self.assertFalse(result.has_prev)

    def test_safe_next_page_with_last_seen(self):
        result1 = self.paginator.paginate_by_page(
            page=1, page_size=10, sort_by="id", safe_mode=True
        )
        result2 = self.paginator.paginate_by_page(
            page=2, page_size=10, sort_by="id",
            safe_mode=True, last_seen_value=result1.last_seen_value, direction="next"
        )
        self.assertEqual(len(result2.data), 10)
        self.assertEqual(result2.data["id"].iloc[0], 11)
        self.assertEqual(result2.data["id"].iloc[-1], 20)
        self.assertEqual(result2.first_seen_value, 11)
        self.assertEqual(result2.last_seen_value, 20)
        self.assertTrue(result2.has_next)
        self.assertTrue(result2.has_prev)

    def test_safe_prev_page_with_first_seen(self):
        result1 = self.paginator.paginate_by_page(
            page=1, page_size=10, sort_by="id", safe_mode=True
        )
        result2 = self.paginator.paginate_by_page(
            page=2, page_size=10, sort_by="id",
            safe_mode=True, last_seen_value=result1.last_seen_value, direction="next"
        )
        result3 = self.paginator.paginate_by_page(
            page=1, page_size=10, sort_by="id",
            safe_mode=True, first_seen_value=result2.first_seen_value, direction="prev"
        )
        self.assertEqual(len(result3.data), 10)
        self.assertEqual(result3.data["id"].iloc[0], 1)
        self.assertEqual(result3.data["id"].iloc[-1], 10)
        self.assertFalse(result3.has_prev)

    def test_safe_last_page(self):
        result = self.paginator.paginate_by_page(
            page=3, page_size=10, sort_by="id",
            safe_mode=True, last_seen_value=20, direction="next"
        )
        self.assertEqual(len(result.data), 10)
        self.assertEqual(result.data["id"].iloc[0], 21)
        self.assertEqual(result.data["id"].iloc[-1], 30)
        self.assertFalse(result.has_next)
        self.assertIsNone(result.last_seen_value)

    def test_safe_traverse_all_pages_forward(self):
        all_ids = []
        last_seen = None
        page = 1
        while True:
            result = self.paginator.paginate_by_page(
                page=page, page_size=10, sort_by="id",
                safe_mode=True, last_seen_value=last_seen, direction="next"
            )
            all_ids.extend(result.data["id"].tolist())
            if not result.has_next:
                break
            last_seen = result.last_seen_value
            page += 1
        self.assertEqual(all_ids, list(range(1, 31)))

    def test_safe_mode_dict_output(self):
        result = self.paginator.paginate_by_page(
            page=1, page_size=5, sort_by="id", safe_mode=True
        )
        d = result.to_dict()
        self.assertTrue(d["safe_mode"])
        self.assertEqual(d["last_seen_value"], 5)
        self.assertEqual(d["first_seen_value"], 1)


class TestDataMutationBugFix(unittest.TestCase):
    def test_normal_pagination_skips_on_delete(self):
        df = pd.DataFrame({"id": range(1, 21)})
        paginator = DataFramePaginator(df)

        page1 = paginator.paginate_by_page(page=1, page_size=5, sort_by="id")
        self.assertEqual(page1.data["id"].tolist(), [1, 2, 3, 4, 5])

        df_after_delete = df[df["id"] != 3].reset_index(drop=True)
        paginator2 = DataFramePaginator(df_after_delete)

        page2 = paginator2.paginate_by_page(page=2, page_size=5, sort_by="id")
        self.assertNotIn(3, page2.data["id"].tolist())
        self.assertEqual(page2.data["id"].tolist(), [7, 8, 9, 10, 11])
        self.assertNotIn(6, page2.data["id"].tolist())

    def test_safe_pagination_no_skip_on_delete(self):
        df = pd.DataFrame({"id": range(1, 21)})
        paginator = DataFramePaginator(df)

        page1 = paginator.paginate_by_page(
            page=1, page_size=5, sort_by="id", safe_mode=True
        )
        self.assertEqual(page1.data["id"].tolist(), [1, 2, 3, 4, 5])

        df_after_delete = df[df["id"] != 3].reset_index(drop=True)
        paginator2 = DataFramePaginator(df_after_delete)

        page2 = paginator2.paginate_by_page(
            page=2, page_size=5, sort_by="id",
            safe_mode=True, last_seen_value=page1.last_seen_value, direction="next"
        )
        self.assertEqual(page2.data["id"].tolist(), [6, 7, 8, 9, 10])
        self.assertNotIn(3, page2.data["id"].tolist())
        self.assertIn(6, page2.data["id"].tolist())

    def test_normal_pagination_duplicates_on_insert(self):
        df = pd.DataFrame({"id": [1, 2, 4, 5, 6, 7, 8, 9, 10]})
        paginator = DataFramePaginator(df)

        page1 = paginator.paginate_by_page(page=1, page_size=5, sort_by="id")
        self.assertEqual(page1.data["id"].tolist(), [1, 2, 4, 5, 6])

        df_after_insert = pd.concat([
            pd.DataFrame({"id": [3]}),
            df
        ]).sort_values("id").reset_index(drop=True)
        paginator2 = DataFramePaginator(df_after_insert)

        page2 = paginator2.paginate_by_page(page=2, page_size=5, sort_by="id")
        self.assertEqual(page2.data["id"].tolist(), [6, 7, 8, 9, 10])
        self.assertIn(6, page1.data["id"].tolist())
        self.assertIn(6, page2.data["id"].tolist())

    def test_safe_pagination_no_duplicate_on_insert(self):
        df = pd.DataFrame({"id": [1, 2, 4, 5, 6, 7, 8, 9, 10]})
        paginator = DataFramePaginator(df)

        page1 = paginator.paginate_by_page(
            page=1, page_size=5, sort_by="id", safe_mode=True
        )
        self.assertEqual(page1.data["id"].tolist(), [1, 2, 4, 5, 6])

        df_after_insert = pd.concat([
            pd.DataFrame({"id": [3]}),
            df
        ]).sort_values("id").reset_index(drop=True)
        paginator2 = DataFramePaginator(df_after_insert)

        page2 = paginator2.paginate_by_page(
            page=2, page_size=5, sort_by="id",
            safe_mode=True, last_seen_value=page1.last_seen_value, direction="next"
        )
        self.assertEqual(page2.data["id"].tolist(), [7, 8, 9, 10])
        self.assertIn(6, page1.data["id"].tolist())
        self.assertNotIn(6, page2.data["id"].tolist())

    def test_safe_pagination_descending_on_delete(self):
        df = pd.DataFrame({"id": range(1, 21)})
        paginator = DataFramePaginator(df)

        page1 = paginator.paginate_by_page(
            page=1, page_size=5, sort_by="id", ascending=False, safe_mode=True
        )
        self.assertEqual(page1.data["id"].tolist(), [20, 19, 18, 17, 16])

        df_after_delete = df[df["id"] != 18].reset_index(drop=True)
        paginator2 = DataFramePaginator(df_after_delete)

        page2 = paginator2.paginate_by_page(
            page=2, page_size=5, sort_by="id",
            ascending=False, safe_mode=True,
            last_seen_value=page1.last_seen_value, direction="next"
        )
        self.assertEqual(page2.data["id"].tolist(), [15, 14, 13, 12, 11])
        self.assertNotIn(18, page2.data["id"].tolist())
        self.assertIn(15, page2.data["id"].tolist())

    def test_cursor_pagination_no_skip_on_delete(self):
        df = pd.DataFrame({"id": range(1, 21)})
        paginator = DataFramePaginator(df)

        page1 = paginator.paginate_by_cursor(cursor=None, limit=5, sort_by="id")
        self.assertEqual(page1.data["id"].tolist(), [1, 2, 3, 4, 5])

        df_after_delete = df[df["id"] != 3].reset_index(drop=True)
        paginator2 = DataFramePaginator(df_after_delete)

        page2 = paginator2.paginate_by_cursor(
            cursor=page1.next_cursor, limit=5, sort_by="id", direction="next"
        )
        self.assertEqual(page2.data["id"].tolist(), [6, 7, 8, 9, 10])
        self.assertNotIn(3, page2.data["id"].tolist())
        self.assertIn(6, page2.data["id"].tolist())


class TestCountTotal(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": range(1, 101)})
        self.paginator = DataFramePaginator(self.df)

    def test_page_mode_always_returns_total(self):
        result = self.paginator.paginate_by_page(page=1, page_size=10, sort_by="id")
        self.assertIsNotNone(result.total)
        self.assertEqual(result.total, 100)

    def test_page_safe_mode_always_returns_total(self):
        result = self.paginator.paginate_by_page(
            page=1, page_size=10, sort_by="id", safe_mode=True
        )
        self.assertIsNotNone(result.total)
        self.assertEqual(result.total, 100)

    def test_cursor_default_no_total(self):
        result = self.paginator.paginate_by_cursor(cursor=None, limit=10, sort_by="id")
        self.assertIsNone(result.total)

    def test_cursor_count_total_true(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=True
        )
        self.assertIsNotNone(result.total)
        self.assertEqual(result.total, 100)

    def test_cursor_count_total_false_explicit(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=False
        )
        self.assertIsNone(result.total)

    def test_cursor_count_total_with_next_page(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=True
        )
        self.assertEqual(result.total, 100)

        result2 = self.paginator.paginate_by_cursor(
            cursor=result.next_cursor, limit=10, sort_by="id",
            direction="next", count_total=True
        )
        self.assertEqual(result2.total, 100)

    def test_cursor_count_total_with_prev_page(self):
        result = self.paginator.paginate_by_cursor(
            cursor=11, limit=10, sort_by="id",
            direction="prev", count_total=True
        )
        self.assertEqual(result.total, 100)

    def test_cursor_count_total_dict_output(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=True
        )
        d = result.to_dict()
        self.assertIn("total", d)
        self.assertEqual(d["total"], 100)

    def test_cursor_no_count_total_dict_output(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=False
        )
        d = result.to_dict()
        self.assertNotIn("total", d)

    def test_page_mode_dict_always_has_total(self):
        result = self.paginator.paginate_by_page(page=1, page_size=10, sort_by="id")
        d = result.to_dict()
        self.assertIn("total", d)
        self.assertEqual(d["total"], 100)

    def test_unified_paginate_cursor_count_total(self):
        result = self.paginator.paginate(
            mode="cursor", limit=10, sort_by="id", count_total=True
        )
        self.assertEqual(result.total, 100)

    def test_unified_paginate_cursor_no_count_total(self):
        result = self.paginator.paginate(
            mode="cursor", limit=10, sort_by="id", count_total=False
        )
        self.assertIsNone(result.total)

    def test_unified_paginate_page_ignores_count_total(self):
        result = self.paginator.paginate(
            mode="page", page=1, page_size=10, sort_by="id", count_total=False
        )
        self.assertIsNotNone(result.total)
        self.assertEqual(result.total, 100)

    def test_cursor_count_total_descending(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id",
            ascending=False, count_total=True
        )
        self.assertEqual(result.total, 100)

    def test_repr_with_no_total(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=False
        )
        self.assertIn("N/A", repr(result))

    def test_repr_with_total(self):
        result = self.paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=True
        )
        self.assertIn("100", repr(result))

    def test_empty_dataframe_cursor_count_total(self):
        df = pd.DataFrame({"id": []})
        paginator = DataFramePaginator(df)
        result = paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=True
        )
        self.assertEqual(result.total, 0)

    def test_empty_dataframe_cursor_no_count_total(self):
        df = pd.DataFrame({"id": []})
        paginator = DataFramePaginator(df)
        result = paginator.paginate_by_cursor(
            cursor=None, limit=10, sort_by="id", count_total=False
        )
        self.assertIsNone(result.total)


if __name__ == "__main__":
    unittest.main()
