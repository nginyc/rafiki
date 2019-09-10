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
          Displays the datasets in list form
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'LIST_DATASET'}
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
          {'ustore> list_dataset\n[SUCCESS: LIST_DATASET] Datasets: []'}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
