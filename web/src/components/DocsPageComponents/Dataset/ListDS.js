import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { gruvboxDark, solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';


const styles = {
  card: {
    maxWidth: "100%",
  },
};

function DocsCard(props) {
  const { classes } = props;
  return (
    <Card className={classes.card}>
      <CardContent>
        <Typography gutterBottom variant="h3" component="h1">
          List Dataset
        </Typography>
        <Typography component="p">
          Display the datasets in list form, maximum number of datasets displayed is 20. To list out all the datasets in the storage, use "LIST_DATASET_ALL"
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'LIST_DATASET{_ALL}'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'// none'}
        </SyntaxHighlighter>
        <Typography component="p">
          Utility Options:
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'// (it is one, not "l") list one entry per line \n-1 [ --vert-list ]'}
        </SyntaxHighlighter>
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
ustore> list_dataset
[SUCCESS: LIST_DATASET] Datasets: []

ustore> list_dataset
[SUCCESS: LIST_DATASET] Datasets: ["DS2", "sampleDS1"]

ustore> list_dataset -1
DS2  ["master"]
sampleDS1  ["master", "newFeature"]

// -1 is equal to --vert-list
ustore> list_dataset --vert-list
DS2  ["master"]
sampleDS1  ["master", "newFeature"]
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
