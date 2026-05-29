# Mermaid Diagram Reference

## Colour Palette (classDefs)

Apply these to any `flowchart` or `graph` diagram. Each colour has two variants: dark text (`color:#000`) and white text (`color:#FFF`, suffixed with `white`).

```
classDef red fill:#ff6666,stroke:#ff0000,color:#000
classDef redwhite fill:#ff6666,stroke:#ff0000,color:#FFF
classDef lightred fill:#ffcccb,stroke:#ff0000,color:#000
classDef lightredwhite fill:#ffcccb,stroke:#ff0000,color:#FFF
classDef orange fill:#ffaa66,stroke:#ff6600,color:#000
classDef orangewhite fill:#ffaa66,stroke:#ff6600,color:#FFF
classDef yellow fill:#ffff66,stroke:#ffcc00,color:#000
classDef yellowwhite fill:#ffff66,stroke:#ffcc00,color:#FFF
classDef green fill:#99ff99,stroke:#00ff00,color:#000
classDef greenwhite fill:#99ff99,stroke:#00ff00,color:#FFF
classDef teal fill:#66ffee,stroke:#00ccaa,color:#000
classDef tealwhite fill:#66ffee,stroke:#00ccaa,color:#FFF
classDef blue fill:#2196F3,stroke:#0066cc,color:#000
classDef bluewhite fill:#2196F3,stroke:#0066cc,color:#FFF
classDef indigo fill:#6666ff,stroke:#3300cc,color:#000
classDef indigowhite fill:#6666ff,stroke:#3300cc,color:#FFF
classDef purple fill:#cc66ff,stroke:#9900cc,color:#000
classDef purplewhite fill:#cc66ff,stroke:#9900cc,color:#FFF
classDef pink fill:#ff66cc,stroke:#cc0099,color:#000
classDef pinkwhite fill:#ff66cc,stroke:#cc0099,color:#FFF
classDef magenta fill:#ff66ff,stroke:#cc00cc,color:#000
classDef magentawhite fill:#ff66ff,stroke:#cc00cc,color:#FFF
classDef brown fill:#cc9966,stroke:#996633,color:#000
classDef brownwhite fill:#cc9966,stroke:#996633,color:#FFF
classDef lime fill:#ccff66,stroke:#99cc00,color:#000
classDef limewhite fill:#ccff66,stroke:#99cc00,color:#FFF
classDef cyan fill:#66ccff,stroke:#0099cc,color:#000
classDef cyanwhite fill:#66ccff,stroke:#0099cc,color:#FFF
classDef slate fill:#99aacc,stroke:#445577,color:#000
classDef slatewhite fill:#99aacc,stroke:#445577,color:#FFF
classDef grey fill:#cccccc,stroke:#888888,color:#000
classDef greywhite fill:#cccccc,stroke:#888888,color:#FFF
classDef black fill:#444444,stroke:#000000,color:#FFF
classDef blackwhite fill:#444444,stroke:#000000,color:#FFF
```

### Palette quick-reference

| Name      | Fill      | Stroke    | Use `white` variant when... |
|-----------|-----------|-----------|-----------------------------|
| red       | #ff6666   | #ff0000   | background is dark          |
| lightred  | #ffcccb   | #ff0000   | softer red needed           |
| orange    | #ffaa66   | #ff6600   |                             |
| yellow    | #ffff66   | #ffcc00   |                             |
| green     | #99ff99   | #00ff00   |                             |
| teal      | #66ffee   | #00ccaa   |                             |
| blue      | #2196F3   | #0066cc   | white text often needed     |
| indigo    | #6666ff   | #3300cc   | white text often needed     |
| purple    | #cc66ff   | #9900cc   |                             |
| pink      | #ff66cc   | #cc0099   |                             |
| magenta   | #ff66ff   | #cc00cc   |                             |
| brown     | #cc9966   | #996633   |                             |
| lime      | #ccff66   | #99cc00   |                             |
| cyan      | #66ccff   | #0099cc   |                             |
| slate     | #99aacc   | #445577   |                             |
| grey      | #cccccc   | #888888   |                             |
| black     | #444444   | #000000   | always use white text       |

### Applying classDefs

Assign a class to a node with `:::` or a `class` statement:

```
flowchart LR
    A[Start]:::green --> B[Process]:::blue --> C[End]:::red

    classDef green fill:#99ff99,stroke:#00ff00,color:#000
    classDef blue fill:#2196F3,stroke:#0066cc,color:#FFF
    classDef red fill:#ff6666,stroke:#ff0000,color:#000
```

Or apply to multiple nodes at once:

```
class A,B green
class C,D bluewhite
```
