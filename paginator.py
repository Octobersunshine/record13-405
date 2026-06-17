import pandas as pd
from typing import Any, Optional, Dict


class PageResult:
    def __init__(
        self,
        data: pd.DataFrame,
        total: int,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        total_pages: Optional[int] = None,
        has_next: bool = False,
        has_prev: bool = False,
        next_cursor: Optional[Any] = None,
        prev_cursor: Optional[Any] = None,
        cursor: Optional[Any] = None,
        limit: Optional[int] = None,
        last_seen_value: Optional[Any] = None,
        first_seen_value: Optional[Any] = None,
        safe_mode: bool = False,
    ):
        self.data = data
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = total_pages
        self.has_next = has_next
        self.has_prev = has_prev
        self.next_cursor = next_cursor
        self.prev_cursor = prev_cursor
        self.cursor = cursor
        self.limit = limit
        self.last_seen_value = last_seen_value
        self.first_seen_value = first_seen_value
        self.safe_mode = safe_mode

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data": self.data.to_dict(orient="records"),
            "total": self.total,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
        }
        if self.page is not None:
            result["page"] = self.page
            result["page_size"] = self.page_size
            result["total_pages"] = self.total_pages
        if self.safe_mode:
            result["safe_mode"] = True
            result["last_seen_value"] = self.last_seen_value
            result["first_seen_value"] = self.first_seen_value
        if self.cursor is not None or self.next_cursor is not None:
            result["cursor"] = self.cursor
            result["limit"] = self.limit
            result["next_cursor"] = self.next_cursor
            result["prev_cursor"] = self.prev_cursor
        return result

    def __repr__(self) -> str:
        mode = "page" if self.page is not None else "cursor"
        if self.safe_mode:
            mode = "safe_page"
        return f"PageResult(mode={mode}, total={self.total}, rows={len(self.data)})"


class DataFramePaginator:
    def __init__(self, df: pd.DataFrame):
        if df is None:
            raise ValueError("DataFrame cannot be None")
        self.df = df.copy()
        self._original_index_name = self.df.index.name

    def paginate_by_page(
        self,
        page: int = 1,
        page_size: int = 10,
        sort_by: Optional[str] = None,
        ascending: bool = True,
        safe_mode: bool = False,
        last_seen_value: Optional[Any] = None,
        first_seen_value: Optional[Any] = None,
        direction: str = "next",
    ) -> PageResult:
        if page < 1:
            raise ValueError("page must be >= 1")
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        if direction not in ("next", "prev"):
            raise ValueError("direction must be 'next' or 'prev'")

        df = self.df.copy()

        if sort_by:
            df = df.sort_values(by=sort_by, ascending=ascending)

        total = len(df)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        if safe_mode:
            if sort_by is None:
                raise ValueError("sort_by is required for safe_mode pagination")

            return self._paginate_by_page_safe(
                df=df,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                ascending=ascending,
                total=total,
                total_pages=total_pages,
                last_seen_value=last_seen_value,
                first_seen_value=first_seen_value,
                direction=direction,
            )

        start = (page - 1) * page_size
        end = start + page_size
        page_data = df.iloc[start:end].copy()

        has_next = page < total_pages
        has_prev = page > 1

        return PageResult(
            data=page_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
        )

    def _paginate_by_page_safe(
        self,
        df: pd.DataFrame,
        page: int,
        page_size: int,
        sort_by: str,
        ascending: bool,
        total: int,
        total_pages: int,
        last_seen_value: Optional[Any],
        first_seen_value: Optional[Any],
        direction: str,
    ) -> PageResult:
        if direction == "next":
            if last_seen_value is not None:
                if ascending:
                    mask = df[sort_by] > last_seen_value
                else:
                    mask = df[sort_by] < last_seen_value
                filtered = df[mask]
                page_data = filtered.head(page_size).copy()

                has_next = len(filtered) > page_size
                has_prev = True
            else:
                filtered = df
                page_data = df.head(page_size).copy()

                has_next = len(df) > page_size
                has_prev = False

            current_last_seen = self._get_last_value(page_data, sort_by) if has_next else None
            current_first_seen = self._get_first_value(page_data, sort_by)
        else:
            if first_seen_value is not None:
                if ascending:
                    mask = df[sort_by] < first_seen_value
                else:
                    mask = df[sort_by] > first_seen_value
                filtered = df[mask]
                page_data = filtered.tail(page_size).copy()

                has_next = True
                has_prev = len(filtered) > page_size
            else:
                filtered = df
                page_data = df.tail(page_size).copy()

                has_next = False
                has_prev = len(df) > page_size

            current_last_seen = self._get_last_value(page_data, sort_by)
            current_first_seen = self._get_first_value(page_data, sort_by) if has_prev else None

        return PageResult(
            data=page_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
            last_seen_value=current_last_seen,
            first_seen_value=current_first_seen,
            safe_mode=True,
        )

    def paginate_by_cursor(
        self,
        cursor: Optional[Any] = None,
        limit: int = 10,
        sort_by: str = "id",
        ascending: bool = True,
        direction: str = "next",
    ) -> PageResult:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if direction not in ("next", "prev"):
            raise ValueError("direction must be 'next' or 'prev'")
        if sort_by not in self.df.columns:
            raise ValueError(f"sort_by column '{sort_by}' does not exist in DataFrame")

        df = self.df.copy()
        df = df.sort_values(by=sort_by, ascending=ascending)

        total = len(df)

        if cursor is None:
            page_data = df.head(limit).copy()
            has_prev = False
            has_next = len(df) > limit
            next_cursor = self._get_last_value(page_data, sort_by) if has_next else None
            prev_cursor = None
            current_cursor = None
        else:
            if direction == "next":
                if ascending:
                    mask = df[sort_by] > cursor
                else:
                    mask = df[sort_by] < cursor
                filtered = df[mask]
                page_data = filtered.head(limit).copy()

                has_next = len(filtered) > limit
                has_prev = True

                next_cursor = self._get_last_value(page_data, sort_by) if has_next else None
                prev_cursor = self._get_first_value(page_data, sort_by)
                current_cursor = cursor
            else:
                if ascending:
                    mask = df[sort_by] < cursor
                else:
                    mask = df[sort_by] > cursor
                filtered = df[mask]

                page_data = filtered.tail(limit).copy()

                has_next = True
                has_prev = len(filtered) > limit

                next_cursor = self._get_last_value(page_data, sort_by)
                prev_cursor = self._get_first_value(page_data, sort_by) if has_prev else None
                current_cursor = cursor

        return PageResult(
            data=page_data,
            total=total,
            cursor=current_cursor,
            limit=limit,
            has_next=has_next,
            has_prev=has_prev,
            next_cursor=next_cursor,
            prev_cursor=prev_cursor,
        )

    @staticmethod
    def _get_last_value(df: pd.DataFrame, column: str) -> Optional[Any]:
        if len(df) == 0:
            return None
        return df[column].iloc[-1]

    @staticmethod
    def _get_first_value(df: pd.DataFrame, column: str) -> Optional[Any]:
        if len(df) == 0:
            return None
        return df[column].iloc[0]

    def paginate(
        self,
        mode: str = "page",
        page: int = 1,
        page_size: int = 10,
        cursor: Optional[Any] = None,
        limit: int = 10,
        sort_by: Optional[str] = None,
        ascending: bool = True,
        direction: str = "next",
        safe_mode: bool = False,
        last_seen_value: Optional[Any] = None,
        first_seen_value: Optional[Any] = None,
    ) -> PageResult:
        if mode == "page":
            return self.paginate_by_page(
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                ascending=ascending,
                safe_mode=safe_mode,
                last_seen_value=last_seen_value,
                first_seen_value=first_seen_value,
                direction=direction,
            )
        elif mode == "cursor":
            if sort_by is None:
                raise ValueError("sort_by is required for cursor pagination")
            return self.paginate_by_cursor(
                cursor=cursor,
                limit=limit,
                sort_by=sort_by,
                ascending=ascending,
                direction=direction,
            )
        else:
            raise ValueError(f"Unknown pagination mode: {mode}. Use 'page' or 'cursor'.")
