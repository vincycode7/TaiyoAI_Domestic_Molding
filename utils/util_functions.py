import plotly.express as px
import pandas as pd

import pandas as pd
import plotly.express as px

import plotly.express as px
import pandas as pd

def target_correlation_chart(X, Y):
    # Combine X and Y into a single dataframe
    combined_df = pd.concat([X, Y], axis=1)

    # Calculate the correlation between Y and each feature
    target_corr = combined_df.corr()[Y.columns]

    # Create a correlation heatmap
    fig = px.imshow(target_corr[:-len(target_corr.columns)], color_continuous_scale='plasma')
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Target",
        font=dict(size=14),
        xaxis_tickangle=-45,
        yaxis_tickangle=0,
    )
    # Show the plot
    fig.show()
    
    return pd.DataFrame(target_corr)