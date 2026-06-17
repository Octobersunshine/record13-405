import pandas as pd
import numpy as np
from paginator import DataFramePaginator


def generate_sample_data(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "id": range(1, n + 1),
        "name": [f"user_{i}" for i in range(1, n + 1)],
        "age": np.random.randint(18, 65, n),
        "score": np.round(np.random.uniform(60, 100, n), 2),
        "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR", "Finance"], n),
    }
    return pd.DataFrame(data)


def demo_page_pagination():
    print("=" * 60)
    print(" 跳页分页演示 (Page/PageSize)")
    print("=" * 60)

    df = generate_sample_data(50)
    paginator = DataFramePaginator(df)

    print(f"\n总数据量: {len(df)} 条")
    print(f"每页大小: 10 条")
    print(f"总页数: {(len(df) + 9) // 10} 页")

    print("\n--- 第 1 页 ---")
    result = paginator.paginate_by_page(page=1, page_size=10, sort_by="id", ascending=True)
    print(f"当前页: {result.page} / {result.total_pages}")
    print(f"数据量: {len(result.data)} 条")
    print(f"有上一页: {result.has_prev}, 有下一页: {result.has_next}")
    print(result.data[["id", "name", "age"]].to_string(index=False))

    print("\n--- 第 3 页 ---")
    result = paginator.paginate_by_page(page=3, page_size=10, sort_by="id", ascending=True)
    print(f"当前页: {result.page} / {result.total_pages}")
    print(f"数据量: {len(result.data)} 条")
    print(f"有上一页: {result.has_prev}, 有下一页: {result.has_next}")
    print(result.data[["id", "name", "age"]].to_string(index=False))

    print("\n--- 最后一页 (第 5 页) ---")
    result = paginator.paginate_by_page(page=5, page_size=10, sort_by="id", ascending=True)
    print(f"当前页: {result.page} / {result.total_pages}")
    print(f"数据量: {len(result.data)} 条")
    print(f"有上一页: {result.has_prev}, 有下一页: {result.has_next}")
    print(result.data[["id", "name", "age"]].to_string(index=False))

    print("\n--- 按分数降序排列，第 2 页 ---")
    result = paginator.paginate_by_page(page=2, page_size=5, sort_by="score", ascending=False)
    print(f"当前页: {result.page} / {result.total_pages}")
    print(result.data[["id", "name", "score", "department"]].to_string(index=False))


def demo_cursor_pagination():
    print("\n" + "=" * 60)
    print(" 游标分页演示 (Cursor/Limit)")
    print("=" * 60)

    df = generate_sample_data(50)
    paginator = DataFramePaginator(df)

    print(f"\n总数据量: {len(df)} 条")

    print("\n--- 第 1 页 (无前驱游标) ---")
    result = paginator.paginate_by_cursor(cursor=None, limit=10, sort_by="id", ascending=True)
    print(f"当前游标: {result.cursor}")
    print(f"下一页游标: {result.next_cursor}")
    print(f"上一页游标: {result.prev_cursor}")
    print(f"有上一页: {result.has_prev}, 有下一页: {result.has_next}")
    print(result.data[["id", "name", "age"]].to_string(index=False))

    first_next_cursor = result.next_cursor

    print("\n--- 第 2 页 (使用 next_cursor 前进) ---")
    result = paginator.paginate_by_cursor(
        cursor=first_next_cursor, limit=10, sort_by="id", ascending=True, direction="next"
    )
    print(f"当前游标: {result.cursor}")
    print(f"下一页游标: {result.next_cursor}")
    print(f"上一页游标: {result.prev_cursor}")
    print(f"有上一页: {result.has_prev}, 有下一页: {result.has_next}")
    print(result.data[["id", "name", "age"]].to_string(index=False))

    second_prev_cursor = result.prev_cursor
    second_next_cursor = result.next_cursor

    print("\n--- 返回上一页 (使用 prev_cursor 后退) ---")
    result = paginator.paginate_by_cursor(
        cursor=second_prev_cursor, limit=10, sort_by="id", ascending=True, direction="prev"
    )
    print(f"当前游标: {result.cursor}")
    print(f"下一页游标: {result.next_cursor}")
    print(f"上一页游标: {result.prev_cursor}")
    print(f"有上一页: {result.has_prev}, 有下一页: {result.has_next}")
    print(result.data[["id", "name", "age"]].to_string(index=False))

    print("\n--- 按分数降序的游标分页 ---")
    result = paginator.paginate_by_cursor(
        cursor=None, limit=5, sort_by="score", ascending=False
    )
    print(f"当前游标: {result.cursor}")
    print(f"下一页游标: {result.next_cursor}")
    print(result.data[["id", "name", "score", "department"]].to_string(index=False))

    if result.has_next:
        print("\n--- 分数降序第 2 页 ---")
        result = paginator.paginate_by_cursor(
            cursor=result.next_cursor, limit=5, sort_by="score", ascending=False, direction="next"
        )
        print(f"当前游标: {result.cursor}")
        print(f"下一页游标: {result.next_cursor}")
        print(f"上一页游标: {result.prev_cursor}")
        print(result.data[["id", "name", "score", "department"]].to_string(index=False))


def demo_unified_api():
    print("\n" + "=" * 60)
    print(" 统一 API 演示 (paginate 方法)")
    print("=" * 60)

    df = generate_sample_data(30)
    paginator = DataFramePaginator(df)

    print("\n--- 使用 mode='page' ---")
    result = paginator.paginate(mode="page", page=2, page_size=5, sort_by="id")
    print(result)
    print(f"第 {result.page} 页，共 {result.total_pages} 页")

    print("\n--- 使用 mode='cursor' ---")
    result = paginator.paginate(mode="cursor", cursor=None, limit=5, sort_by="id")
    print(result)
    print(f"下一页游标: {result.next_cursor}")

    print("\n--- 转换为字典格式（便于 API 返回） ---")
    result = paginator.paginate(mode="page", page=1, page_size=3, sort_by="id")
    result_dict = result.to_dict()
    print(f"keys: {list(result_dict.keys())}")
    print(f"total: {result_dict['total']}")
    print(f"page: {result_dict['page']}")
    print(f"data 条数: {len(result_dict['data'])}")
    print(f"第一条数据: {result_dict['data'][0]}")


def demo_large_dataset():
    print("\n" + "=" * 60)
    print(" 大数据量性能演示 (10万条)")
    print("=" * 60)

    import time

    n = 100_000
    print(f"\n生成 {n:,} 条数据...")
    df = generate_sample_data(n)
    paginator = DataFramePaginator(df)

    print("\n--- 跳页分页：访问第 5000 页 ---")
    start = time.time()
    result = paginator.paginate_by_page(page=5000, page_size=20, sort_by="id")
    elapsed = time.time() - start
    print(f"耗时: {elapsed:.4f} 秒")
    print(f"第 {result.page} 页，共 {result.total_pages} 页")
    print(f"数据条数: {len(result.data)}")

    print("\n--- 游标分页：从中间位置开始 ---")
    mid_cursor = 50_000
    start = time.time()
    result = paginator.paginate_by_cursor(cursor=mid_cursor, limit=20, sort_by="id")
    elapsed = time.time() - start
    print(f"耗时: {elapsed:.4f} 秒")
    print(f"数据条数: {len(result.data)}")
    print(f"起始 ID: {result.data['id'].iloc[0] if len(result.data) > 0 else 'N/A'}")


if __name__ == "__main__":
    demo_page_pagination()
    demo_cursor_pagination()
    demo_unified_api()
    demo_large_dataset()

    print("\n" + "=" * 60)
    print(" 数据变动场景对比演示")
    print("=" * 60)

    df = pd.DataFrame({"id": range(1, 21), "name": [f"user_{i}" for i in range(1, 21)]})
    paginator = DataFramePaginator(df)

    print("\n--- 初始数据: ID 1-20 ---")
    print(f"第 1 页请求 (page=1, page_size=5)...")
    page1_normal = paginator.paginate_by_page(page=1, page_size=5, sort_by="id")
    page1_safe = paginator.paginate_by_page(page=1, page_size=5, sort_by="id", safe_mode=True)
    print(f"普通模式第 1 页 ID: {page1_normal.data['id'].tolist()}")
    print(f"安全模式第 1 页 ID: {page1_safe.data['id'].tolist()}")

    print("\n--- 数据变动: 删除 ID=3 ---")
    df_after = df[df["id"] != 3].reset_index(drop=True)
    paginator_after = DataFramePaginator(df_after)

    print("\n--- 请求第 2 页 ---")
    page2_normal = paginator_after.paginate_by_page(page=2, page_size=5, sort_by="id")
    page2_safe = paginator_after.paginate_by_page(
        page=2, page_size=5, sort_by="id",
        safe_mode=True, last_seen_value=page1_safe.last_seen_value, direction="next"
    )

    print(f"普通模式第 2 页 ID: {page2_normal.data['id'].tolist()}")
    print(f"  ❌ 问题: ID=6 被遗漏了! (因为删除导致偏移错误)")
    print(f"安全模式第 2 页 ID: {page2_safe.data['id'].tolist()}")
    print(f"  ✅ 正确: 从 ID=6 开始，基于 last_seen_value=5 定位")

    print("\n--- 使用游标分页也可以解决 ---")
    cursor_page1 = paginator.paginate_by_cursor(cursor=None, limit=5, sort_by="id")
    cursor_page2 = paginator_after.paginate_by_cursor(
        cursor=cursor_page1.next_cursor, limit=5, sort_by="id", direction="next"
    )
    print(f"游标模式第 2 页 ID: {cursor_page2.data['id'].tolist()}")
    print(f"  ✅ 正确: 基于 cursor=5 定位")

    print("\n" + "=" * 60)
    print(" 建议")
    print("=" * 60)
    print("""
  📌 数据频繁变动场景，推荐使用:
     1. 游标分页 (cursor/limit) - 最可靠
     2. 安全跳页 (safe_mode=True) - 保留 page 语义

  ⚠️  普通跳页 (page/pageSize) 仅适用于:
     - 数据相对静态
     - 需要随机跳页访问
     - 可容忍少量重复/遗漏
    """)
