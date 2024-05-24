from ortools.graph.python import min_cost_flow
import polars as pl
import pydantic
import yaml


class WeightDocument(pydantic.BaseModel):
    spec: dict[str, dict[str, str] | str]


class WeightLambda:
    def __init__(self, columns: list[str], primary: str, column: dict[str, str] | str) -> None:
        super().__init__()
        self._primary = primary
        if isinstance(column, dict):
            self._choices = column
            if '_' in column:
                self._global = column['_']
                del column['_']
            else:
                self._global = None
        else:
            self._choices = None
            self._global = column

        if self._choices is not None:
            # self._expr = {
            #     key: eval(value, {}, {
            #         'x': pl.col(self._primary),
            #     })
            #     for key, value in self._choices.items()
            # }
            self._expr = None
            for key, expr in self._choices.items():
                if self._expr is None:
                    self._expr = pl.when(pl.col(self._primary) == key).then(
                        eval(expr, {}, {
                            'x': pl.col(self._primary),
                            **{
                                name: pl.col(name)
                                for name in columns
                            }
                        })
                    )
                else:
                    self._expr = self._expr.when(pl.col(self._primary) == key).then(
                        eval(expr, {}, {
                            'x': pl.col(self._primary),
                            **{
                                name: pl.col(name)
                                for name in columns
                            }
                        })
                    )

            if self._global is not None:
                if self._expr is not None:
                    self._expr = self._expr.otherwise(
                        eval(self._global, {}, {
                            'x': pl.col(self._primary),
                            **{
                                name: pl.col(name)
                                for name in columns
                            }
                        })
                    )
                else:
                    self._expr = eval(self._global, {}, {
                        'x': pl.col(self._primary),
                        **{
                            name: pl.col(name)
                            for name in columns
                        }
                    })
            elif self._expr is None:
                self._expr = pl.lit(0).cast(pl.Int64)

        elif self._global is not None:
            self._expr = eval(self._global, {}, {
                'x': pl.col(self._primary),
                **{
                    name: pl.col(name)
                    for name in columns
                }
            })

        else:
            self._expr = pl.lit(0).cast(pl.Int64)

    def __call__(self) -> pl.Expr:
        return self._expr.cast(pl.Int64)


def load_weights(columns: list[str]) -> list[WeightLambda]:
    with open('./data/weights.yaml', 'r', encoding='utf-8') as f:
        raw = yaml.load(f, yaml.SafeLoader)

    doc = WeightDocument.model_validate(raw)
    return [
        WeightLambda(columns, key, value)
        for key, value in doc.spec.items()
    ]


def calculate_penalty(columns: list[str]) -> pl.Expr:
    penalty = pl.lit(0).cast(pl.Int64)

    for weight in load_weights(columns):
        penalty += weight()

    return penalty


def main():
    schedule = pl.read_csv('./data/schedule.csv')
    schedule = schedule.with_columns(
        pl.arange(end=len(schedule)).alias('index'),
    )

    nodes = schedule.with_columns(
        pl.lit(1).cast(pl.Int64).alias('capacity'),
        pl.lit(0).cast(pl.Int64).alias('supply'),
        calculate_penalty(schedule.columns).alias('penalty'),
    )
    print(nodes.select('waypoint', 'cost', 'penalty'))
    node_src = len(nodes)
    node_sink = node_src + 1

    edges_src = nodes \
        .filter(pl.col('direction') == 'in') \
        .select(
            pl.col('direction'),
            pl.lit(node_src).cast(pl.Int64).alias('src'),
            pl.col('index').alias('sink'),
            pl.col('capacity'),
            pl.col('cost'),
            pl.col('penalty'),
        )
    edges_sink = nodes \
        .filter(pl.col('direction') == 'out') \
        .select(
            pl.col('direction'),
            pl.col('index').alias('src'),
            pl.lit(node_sink).cast(pl.Int64).alias('sink'),
            pl.col('capacity'),
            pl.col('cost'),
            pl.col('penalty'),
        )

    nodes_src = nodes.filter(pl.col('direction') == 'in').select(
        pl.col('index').alias('src'),
        pl.col('day').alias('src.day'),
    )
    nodes_sink = nodes.filter(pl.col('direction') == 'out').select(
        pl.col('index').alias('sink'),
        pl.col('day').alias('sink.day'),
    )

    edges_link = nodes_src.join(nodes_sink, how='cross').with_columns(
        pl.lit(1).cast(pl.Int64).alias('capacity'),
        pl.lit(0).cast(pl.Int64).alias('penalty'),
    ).filter(pl.col('src.day') + 9 == pl.col('sink.day')).select(
        pl.lit('stay').alias('direction'),
        pl.col('src'),
        pl.col('sink'),
        pl.col('capacity'),
        pl.lit(0).cast(pl.Int64).alias('cost'),
        pl.col('penalty'),
    )

    edges = pl.concat([edges_src, edges_sink, edges_link])
    edges = edges.with_columns(
        pl.arange(end=len(edges)).alias('index'),
    )

    smcf = min_cost_flow.SimpleMinCostFlow()

    # Add each arc.
    smcf.add_arcs_with_capacity_and_unit_cost(
        edges.get_column('src').to_numpy(),
        edges.get_column('sink').to_numpy(),
        edges.get_column('capacity').to_numpy(),
        edges.get_column('penalty').to_numpy(),
    )

    # Add node supplies.
    smcf.set_nodes_supplies(
        nodes.get_column('index').to_numpy(),
        nodes.get_column('supply').to_numpy(),
    )
    smcf.set_node_supply(node_src, 1)
    smcf.set_node_supply(node_sink, -1)

    # Find the minimum cost flow between node 0 and node 10.
    status = smcf.solve_max_flow_with_min_cost()

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        return

    print("Total cost = ", smcf.optimal_cost())
    edges = edges.with_columns(
        pl.Series(smcf.flows(edges.get_column('index').to_numpy()))
        .alias('flow')
    )
    print(edges.filter((pl.col('penalty') > 0) & (pl.col('flow') > 0)))


if __name__ == '__main__':
    main()
